"""Premise scoring head for unified candidate pools."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .premise_pool import CandidatePool


@dataclass(frozen=True)
class PremiseScorerConfig:
    """Configuration for the premise scoring head."""

    hidden_dim: int = 128
    scoring_mode: str = "dot"  # "dot" or "mlp"
    tactic_conditioning: str = "soft"  # "soft" or "hard"
    premise_loss_weight: float = 0.3
    k: int = 200
    rerank_size: int = 50

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "scoring_mode": self.scoring_mode,
            "tactic_conditioning": self.tactic_conditioning,
            "premise_loss_weight": self.premise_loss_weight,
        }


class PremiseScorer(nn.Module):
    """Score candidates with tactic-conditioned queries. Supports dot or MLP scoring."""

    def __init__(self, hidden_dim: int, *, mode: str = "dot") -> None:
        super().__init__()

        if mode not in {"dot", "mlp"}:
            raise ValueError(f"Unsupported scoring mode '{mode}'. Use 'dot' or 'mlp'.")

        self.mode = mode
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)        # Step 0: [state; tactic]
        self.query_proj_ar = nn.Linear(hidden_dim * 3, hidden_dim)     # Step N: [state; tactic; prev]
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        if mode == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.scorer = None

        self._scale = 1.0 / math.sqrt(hidden_dim)

    def score(
        self,
        state_vec: Tensor,
        tactic_emb: Tensor,
        candidate_vectors: Tensor,
    ) -> Tensor:
        """Score candidates against the proof state."""
        state = state_vec.view(-1)
        tactic = tactic_emb.view(-1)

        query = self.query_proj(torch.cat([state, tactic], dim=0))
        candidate_vectors = self.key_proj(candidate_vectors)

        if self.mode == "dot":
            scores = (candidate_vectors @ query) * self._scale
        else:
            num_candidates = candidate_vectors.size(0)
            query_expanded = query.unsqueeze(0).expand(num_candidates, -1)
            combined = torch.cat([query_expanded, candidate_vectors], dim=1)
            scores = self.scorer(combined).squeeze(-1)

        return scores

    def forward(
        self,
        state_vecs: Tensor,
        tactic_embs: Tensor,
        pools: list[CandidatePool],
    ) -> list[Tensor]:
        """Score all pools in a batch."""
        batch_size = state_vecs.size(0)
        if len(pools) != batch_size:
            raise ValueError(
                f"Number of pools ({len(pools)}) does not match "
                f"batch size ({batch_size})."
            )

        return [
            self.score(state_vecs[b], tactic_embs[b], pools[b].candidate_vectors)
            for b in range(batch_size)
        ]

    def select_arguments(
        self,
        state_vec: Tensor,
        tactic_emb: Tensor,
        candidate_vectors: Tensor,
        num_args: int,
    ) -> tuple[list[int], list[float]]:
        """Autoregressively select arguments. Masks previously selected."""
        num_args = min(num_args, candidate_vectors.size(0))
        if num_args <= 0:
            return [], []

        state = state_vec.view(-1)
        tactic = tactic_emb.view(-1)
        keys = self.key_proj(candidate_vectors)

        selected_indices: list[int] = []
        selected_scores: list[float] = []
        mask = torch.zeros(candidate_vectors.size(0), dtype=torch.bool, device=state.device)

        query = self.query_proj(torch.cat([state, tactic], dim=0))

        for _ in range(num_args):
            if self.mode == "dot":
                raw_scores = (keys @ query) * self._scale
            else:
                num_c = keys.size(0)
                q_exp = query.unsqueeze(0).expand(num_c, -1)
                raw_scores = self.scorer(torch.cat([q_exp, keys], dim=1)).squeeze(-1)

            raw_scores = raw_scores.masked_fill(mask, float("-inf"))
            best = int(raw_scores.argmax().item())
            selected_indices.append(best)
            selected_scores.append(float(raw_scores[best].item()))
            mask[best] = True

            prev = candidate_vectors[best].view(-1)
            query = self.query_proj_ar(torch.cat([state, tactic, prev], dim=0))

        return selected_indices, selected_scores


def _find_target_index_in_pool(
    pool: CandidatePool,
    *,
    arg_node_indices: list[int],
    arg_lemma_ids: list[int],
) -> int:
    """Find true premise in pool. Prefers local over lemma."""
    for node_id in arg_node_indices:
        if node_id < 0:
            continue
        for pool_idx, (source, cid) in enumerate(
            zip(pool.candidate_sources, pool.candidate_ids)
        ):
            if source == "local" and cid == node_id:
                return pool_idx

    for lemma_id in arg_lemma_ids:
        if lemma_id < 0:
            continue
        for pool_idx, (source, cid) in enumerate(
            zip(pool.candidate_sources, pool.candidate_ids)
        ):
            if source == "lemma" and cid == lemma_id:
                return pool_idx

    return -1


def compute_premise_ranking_loss(
    score_list: list[Tensor],
    pools: list[CandidatePool],
    arg_node_indices: Tensor,
    arg_lemma_ids: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Cross-entropy ranking loss over candidate pools with metrics."""
    batch_size = len(score_list)
    device = score_list[0].device if score_list else torch.device("cpu")

    losses: list[Tensor] = []
    valid_count = 0
    target_present_count = 0
    top1_correct = 0
    top5_correct = 0
    mrr_sum = 0.0

    for b in range(batch_size):
        scores = score_list[b]
        pool = pools[b]

        b_node_ids = arg_node_indices[b].tolist() if arg_node_indices.dim() > 1 else [int(arg_node_indices[b].item())]
        b_lemma_ids = arg_lemma_ids[b].tolist() if arg_lemma_ids.dim() > 1 else [int(arg_lemma_ids[b].item())]

        has_target = any(i >= 0 for i in b_node_ids) or any(i >= 0 for i in b_lemma_ids)
        if not has_target:
            continue

        target_present_count += 1
        target_idx = _find_target_index_in_pool(
            pool,
            arg_node_indices=b_node_ids,
            arg_lemma_ids=b_lemma_ids,
        )

        if target_idx < 0:
            continue

        target = torch.tensor(target_idx, dtype=torch.long, device=device)
        loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
        losses.append(loss)
        valid_count += 1

        sorted_indices = scores.argsort(descending=True).tolist()
        rank = sorted_indices.index(target_idx) + 1

        if rank == 1:
            top1_correct += 1
        if rank <= 5:
            top5_correct += 1
        mrr_sum += 1.0 / rank

    if losses:
        total_loss = torch.stack(losses).mean()
    else:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    metrics = {
        "premise_loss": float(total_loss.item()),
        "valid_samples": valid_count,
        "total_samples": batch_size,
        "target_present_count": target_present_count,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "mrr_sum": mrr_sum,
    }

    return total_loss, metrics