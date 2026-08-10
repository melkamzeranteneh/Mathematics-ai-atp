"""Premise-aware training and evaluation loops for PremiseGNN.

These loops train the unified PremiseGNN model with contrastive learning,
replacing the old TacticWithArgsClassifier + PremiseScorer approach.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from .premise_gnn import PremiseGNN
from .premise_pool import build_unified_pools
from .premise_scoring import compute_premise_ranking_loss
from .lemma_index import LemmaIndex
from .reporting import console_print


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def _should_log_batch(
    batch_index: int, total_batches: int, *, log_every_batches: int
) -> bool:
    return (
        batch_index == 1
        or batch_index == total_batches
        or batch_index % log_every_batches == 0
    )


def _extract_arg_targets(
    batch, max_args: int, device: torch.device
) -> torch.Tensor:
    """Extract ground-truth argument node indices [B, max_args], padded with -1."""
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    targets = torch.full((batch_size, max_args), -1, dtype=torch.long, device=device)
    
    if hasattr(batch, "arg_node_indices") and hasattr(batch, "arg_count"):
        flat_targets = batch.arg_node_indices.to(device=device, dtype=torch.long)
        counts = batch.arg_count.tolist()
        offset = 0
        for i, count in enumerate(counts):
            n_copy = min(count, max_args)
            if n_copy > 0:
                targets[i, :n_copy] = flat_targets[offset : offset + n_copy]
            offset += count
            
    return targets


def _extract_arg_lemma_ids(
    batch, max_args: int, device: torch.device
) -> torch.Tensor:
    """Extract ground-truth lemma IDs [B, max_args], padded with -1."""
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    targets = torch.full((batch_size, max_args), -1, dtype=torch.long, device=device)
    
    if hasattr(batch, "arg_lemma_ids") and hasattr(batch, "arg_count"):
        flat_targets = batch.arg_lemma_ids.to(device=device, dtype=torch.long)
        counts = batch.arg_count.tolist()
        offset = 0
        for i, count in enumerate(counts):
            n_copy = min(count, max_args)
            if n_copy > 0:
                targets[i, :n_copy] = flat_targets[offset : offset + n_copy]
            offset += count
            
    return targets


def train_one_epoch_premise_gnn(
    model: PremiseGNN,
    loader: DataLoader,
    lemma_index: LemmaIndex,
    *,
    optimizer: AdamW,
    grad_scaler,
    device: torch.device,
    grad_clip: float,
    temperature: float = 0.07,
    k: int = 500,
    max_args: int = 3,
    epoch: int,
    total_epochs: int,
    log_every_batches: int,
    use_amp: bool,
    pin_memory: bool,
) -> dict[str, float | int]:
    """Train PremiseGNN one epoch with contrastive ranking loss."""
    model.train()

    total_loss = 0.0
    total_examples = 0
    total_batches = len(loader)
    start_time = time.perf_counter()

    console_print(
        f"  Starting epoch {epoch:02d}/{total_epochs:02d} "
        f"with {total_batches} train batches (PremiseGNN contrastive)..."
    )

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(
            device, non_blocking=(device.type == "cuda" and pin_memory)
        )
        tactic_ids = batch.y.view(-1)
        arg_targets = _extract_arg_targets(batch, max_args, device)
        arg_lemma_targets = _extract_arg_lemma_ids(batch, max_args, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Forward pass with full gradient flow
            node_embeddings = model.encode_nodes(batch)
            state_vecs = model.readout(node_embeddings, batch)
            tactic_embs = model.tactic_embedding(tactic_ids)
            
            premise_mask = batch.premise_mask.to(dtype=torch.bool, device=device) if hasattr(batch, "premise_mask") else None
            
            # NO FAISS during training - only in-batch negatives
            pools = build_unified_pools(
                state_vecs,
                node_embeddings,
                premise_mask,
                batch.batch,
                lemma_index=None,
                k=k,
            )
            
            # Score candidates
            score_list = []
            for b in range(len(pools)):
                query = model.query_proj(
                    torch.cat([state_vecs[b], tactic_embs[b]], dim=-1)
                )
                candidate_keys = model.key_proj(pools[b].candidate_vectors)
                scores = (candidate_keys @ query) / temperature
                score_list.append(scores)
            
            # Contrastive ranking loss
            p_loss, p_metrics = compute_premise_ranking_loss(
                score_list,
                pools,
                arg_targets,
                arg_lemma_targets,
            )

        # Backprop - updates ALL parameters
        grad_scaler.scale(p_loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        batch_size = int(tactic_ids.numel())
        total_loss += float(p_loss.item()) * batch_size
        total_examples += batch_size

        if _should_log_batch(
            batch_index, total_batches, log_every_batches=log_every_batches
        ):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            n = max(total_examples, 1)
            console_print(
                f"    train batch {batch_index:>5}/{total_batches} | "
                f"seen={total_examples} | "
                f"loss={total_loss / n:.4f} | "
                f"coverage={p_metrics['target_present_count'] / max(p_metrics['total_samples'], 1):.3f} | "
                f"elapsed={elapsed}"
            )

    n = max(total_examples, 1)
    return {
        "loss": total_loss / n,
        "example_count": total_examples,
        "valid_samples": p_metrics.get("valid_samples", 0),
        "target_present_count": p_metrics.get("target_present_count", 0),
    }


@torch.no_grad()
def evaluate_premise_gnn(
    model: PremiseGNN,
    loader: DataLoader,
    lemma_index: LemmaIndex,
    *,
    device: torch.device,
    temperature: float = 0.07,
    k: int = 500,
    max_args: int = 3,
    split_name: str | None = None,
    log_every_batches: int | None = None,
    use_amp: bool = False,
    pin_memory: bool = False,
) -> dict[str, float | int]:
    """Evaluate PremiseGNN: coverage, MRR, Hit@1/5."""
    model.eval()

    total_loss = 0.0
    total_count = 0
    
    valid_samples = 0
    target_present_count = 0
    mrr_sum = 0.0
    top1_correct = 0
    top5_correct = 0

    total_batches = len(loader)
    start_time = time.perf_counter()

    if split_name is not None:
        console_print(
            f"  Evaluating {split_name} split "
            f"({total_batches} batches, PremiseGNN)..."
        )

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(
            device, non_blocking=(device.type == "cuda" and pin_memory)
        )
        tactic_ids = batch.y.view(-1)
        arg_targets = _extract_arg_targets(batch, max_args, device)
        arg_lemma_targets = _extract_arg_lemma_ids(batch, max_args, device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            node_embeddings = model.encode_nodes(batch)
            state_vecs = model.readout(node_embeddings, batch)
            tactic_embs = model.tactic_embedding(tactic_ids)
            
            premise_mask = batch.premise_mask.to(dtype=torch.bool, device=device) if hasattr(batch, "premise_mask") else None
            
            # Evaluation uses FAISS index (built with final trained weights)
            pools = build_unified_pools(
                state_vecs,
                node_embeddings,
                premise_mask,
                batch.batch,
                lemma_index=lemma_index,
                k=k,
            )
            
            score_list = []
            for b in range(len(pools)):
                query = model.query_proj(
                    torch.cat([state_vecs[b], tactic_embs[b]], dim=-1)
                )
                candidate_keys = model.key_proj(pools[b].candidate_vectors)
                scores = (candidate_keys @ query) / temperature
                score_list.append(scores)
            
            p_loss, p_metrics = compute_premise_ranking_loss(
                score_list, pools, arg_targets, arg_lemma_targets
            )

        bs = int(tactic_ids.numel())
        total_loss += float(p_loss.item()) * bs
        total_count += bs
        
        valid_samples += p_metrics.get("valid_samples", 0)
        target_present_count += p_metrics.get("target_present_count", 0)
        mrr_sum += p_metrics.get("mrr_sum", 0.0)
        top1_correct += p_metrics.get("top1_correct", 0)
        top5_correct += p_metrics.get("top5_correct", 0)

        if (
            split_name is not None
            and log_every_batches is not None
            and _should_log_batch(
                batch_index,
                total_batches,
                log_every_batches=log_every_batches,
            )
        ):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            console_print(
                f"    {split_name} batch {batch_index:>5}/{total_batches} | "
                f"valid={valid_samples} | "
                f"covered={target_present_count} | "
                f"elapsed={elapsed}"
            )

    n = max(total_count, 1)
    valid_n = max(valid_samples, 1)
    
    coverage = target_present_count / max(valid_samples, 1)
    
    return {
        "loss": total_loss / n,
        "coverage": coverage,
        "mrr": mrr_sum / valid_n,
        "hit1": top1_correct / valid_n,
        "hit5": top5_correct / valid_n,
        "valid_samples": valid_samples,
        "target_present_count": target_present_count,
        "evaluated_count": total_count,
    }


# ============================================================================
# DEPRECATED FUNCTIONS - Kept for backward compatibility
# ============================================================================

def train_one_epoch_with_premises(*args, **kwargs):
    """[DEPRECATED] Use train_one_epoch_premise_gnn() instead."""
    raise NotImplementedError("Use train_one_epoch_premise_gnn() with PremiseGNN")


def evaluate_model_with_premises(*args, **kwargs):
    """[DEPRECATED] Use evaluate_premise_gnn() instead."""
    raise NotImplementedError("Use evaluate_premise_gnn() with PremiseGNN")