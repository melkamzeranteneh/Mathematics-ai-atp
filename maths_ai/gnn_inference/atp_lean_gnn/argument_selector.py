"""Pointer-network argument selector for tactic argument prediction.

This module is additive — the existing ``GraphSAGEStateClassifier`` in
``model.py`` is used as a backbone and remains untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .labels import get_tactic_arity
from .model import GraphSAGEStateClassifier
from .pyg import NODE_TYPE_TO_ID


# ---------------------------------------------------------------------------
# ArgumentSelector: Scaled dot-product pointer head
# ---------------------------------------------------------------------------


class ArgumentSelector(nn.Module):
    """Score every node in the DAG as a candidate tactic argument.

    **Query** = concat(state_embedding, tactic_embedding) projected to key-space.
    **Keys**  = node_embeddings from the GNN encoder.

    After selecting argument *k*, the selected node's embedding is fused into
    the query before selecting argument *k + 1* (autoregressive).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # First step: query is [state_emb; tactic_emb] → hidden_dim
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        # Subsequent steps: query is [state_emb; tactic_emb; prev_arg_emb] → hidden_dim
        self.query_proj_ar = nn.Linear(hidden_dim * 3, hidden_dim)
        self._scale = 1.0 / math.sqrt(hidden_dim)

    def forward(
        self,
        state_emb: Tensor,        # [B, H]
        tactic_emb: Tensor,        # [B, H]
        node_embeddings: Tensor,   # [total_nodes, H]
        premise_mask: Tensor,      # [total_nodes]  bool
        batch_index: Tensor,       # [total_nodes]  → which graph each node belongs to
        prev_arg_emb: Tensor | None = None,  # [B, H] or None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(arg_logits, selected_emb)``.

        ``arg_logits``   — shape ``[B, max_nodes_in_batch]``, padded with -inf.
        ``selected_emb`` — shape ``[B, H]``, the embedding of the argmax node
                           (used as context for the next autoregressive step).
        """
        scores, padded_keys = self._score_nodes(
            state_emb, tactic_emb, node_embeddings, premise_mask, batch_index, prev_arg_emb
        )
        batch_size = state_emb.size(0)
        device = state_emb.device
        with torch.no_grad():
            selected_idx = scores.argmax(dim=1)  # [B]
        selected_emb = padded_keys[torch.arange(batch_size, device=device), selected_idx]  # [B, H]
        return scores, selected_emb

    def sample_step(
        self,
        state_emb: Tensor,
        tactic_emb: Tensor,
        node_embeddings: Tensor,
        premise_mask: Tensor,
        batch_index: Tensor,
        prev_arg_emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sampling variant for on-policy RL (B2).

        Return ``(arg_logits, selected_idx, log_prob, selected_emb)`` where ``selected_idx``
        is SAMPLED from ``softmax(arg_logits)`` (not argmax), ``log_prob`` is its log-prob
        under the pointer policy (carries gradient — this is the argument policy gradient),
        and ``selected_emb`` is the sampled node's embedding for the next autoregressive step.
        """
        scores, padded_keys = self._score_nodes(
            state_emb, tactic_emb, node_embeddings, premise_mask, batch_index, prev_arg_emb
        )
        batch_size = state_emb.size(0)
        device = state_emb.device

        dist = torch.distributions.Categorical(logits=scores)
        selected_idx = dist.sample()          # [B]
        log_prob = dist.log_prob(selected_idx)  # [B] — differentiable w.r.t. pointer params
        selected_emb = padded_keys[torch.arange(batch_size, device=device), selected_idx]  # [B, H]
        return scores, selected_idx, log_prob, selected_emb

    def forced_step(
        self,
        state_emb: Tensor,
        tactic_emb: Tensor,
        node_embeddings: Tensor,
        premise_mask: Tensor,
        batch_index: Tensor,
        forced_idx: Tensor,        # [B] long; -1 = no argument at this step for that sample
        prev_arg_emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Teacher-forced variant for the train-phase log-prob recompute.

        The collect phase stores only the sampled argument indices (ints); the train
        phase re-runs the pointer under the CURRENT parameters and evaluates
        ``log π(forced_idx)`` at each step, feeding the stored index's embedding
        forward autoregressively — so the gradient matches the action actually
        executed, without holding the search-time autograd graph.

        Return ``(log_prob, selected_emb)``. Rows with ``forced_idx == -1`` (that
        sample has fewer arguments than this step) or an out-of-range/masked index
        contribute log-prob 0 and a zero embedding.
        """
        scores, padded_keys = self._score_nodes(
            state_emb, tactic_emb, node_embeddings, premise_mask, batch_index, prev_arg_emb
        )
        batch_size = state_emb.size(0)
        device = state_emb.device
        max_nodes = scores.size(1)

        valid = (forced_idx >= 0) & (forced_idx < max_nodes)
        safe_idx = forced_idx.clamp(min=0, max=max_nodes - 1)
        log_probs = torch.log_softmax(scores, dim=1)
        gathered = log_probs.gather(1, safe_idx.unsqueeze(1)).squeeze(1)  # [B]
        # A stored index landing on a masked (-inf) position yields -inf/NaN; drop it too.
        valid = valid & torch.isfinite(gathered)
        log_prob = torch.where(valid, gathered, torch.zeros_like(gathered))

        selected_emb = padded_keys[torch.arange(batch_size, device=device), safe_idx]  # [B, H]
        selected_emb = selected_emb * valid.unsqueeze(1).to(selected_emb.dtype)
        return log_prob, selected_emb

    def _score_nodes(
        self,
        state_emb: Tensor,        # [B, H]
        tactic_emb: Tensor,        # [B, H]
        node_embeddings: Tensor,   # [total_nodes, H]
        premise_mask: Tensor,      # [total_nodes]  bool
        batch_index: Tensor,       # [total_nodes]
        prev_arg_emb: Tensor | None,  # [B, H] or None
    ) -> tuple[Tensor, Tensor]:
        """Shared scoring core: return ``(scores [B, N_max], padded_keys [B, N_max, H])``.

        Non-premise / padding positions in ``scores`` are set to -inf.
        """
        batch_size = state_emb.size(0)
        hidden_dim = state_emb.size(1)
        device = state_emb.device

        # --- 1. Build the query vector --------------------------------
        if prev_arg_emb is None:
            query = self.query_proj(torch.cat([state_emb, tactic_emb], dim=1))  # [B, H]
        else:
            query = self.query_proj_ar(
                torch.cat([state_emb, tactic_emb, prev_arg_emb], dim=1)
            )  # [B, H]

        # --- 2. Scatter node embeddings into a padded [B, N_max, H] tensor -
        #     where N_max is the max number of nodes in any graph in the batch.
        counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        counts.scatter_add_(0, batch_index, torch.ones_like(batch_index))
        max_nodes = int(counts.max().item())

        # Compute per-node offset within its graph
        offsets = torch.zeros_like(batch_index)
        for b in range(batch_size):
            graph_mask = batch_index == b
            offsets[graph_mask] = torch.arange(
                int(graph_mask.sum().item()), device=device, dtype=torch.long
            )

        # Padded node embedding matrix
        padded_keys = torch.zeros(batch_size, max_nodes, hidden_dim, device=device, dtype=node_embeddings.dtype)
        padded_keys[batch_index, offsets] = node_embeddings

        # Padded premise mask (False = invalid ⇒ will be masked to -inf)
        padded_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
        padded_mask[batch_index, offsets] = premise_mask

        # --- 3. Scaled dot-product attention scores -------------------
        # query: [B, H] → [B, 1, H];  keys: [B, N_max, H]
        scores = torch.bmm(query.unsqueeze(1), padded_keys.transpose(1, 2)).squeeze(1)  # [B, N_max]
        scores = scores * self._scale

        # Mask out non-premise positions
        scores = scores.masked_fill(~padded_mask, float("-inf"))

        return scores, padded_keys


# ---------------------------------------------------------------------------
# TacticWithArgsClassifier: Full model wrapping backbone + pointer head
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TacticWithArgsConfig:
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    max_args: int = 3
    arg_loss_weight: float = 0.5

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "max_args": self.max_args,
            "arg_loss_weight": self.arg_loss_weight,
        }


class TacticWithArgsClassifier(nn.Module):
    """Tactic family prediction + pointer-based argument selection.

    The GNN backbone (``GraphSAGEStateClassifier``) is instantiated internally
    and its ``encode_nodes`` / ``readout`` methods are reused.  The tactic
    classification head is inherited from the backbone.  A new
    ``ArgumentSelector`` pointer head is added on top.
    """

    def __init__(
        self,
        *,
        num_node_labels: int,
        num_tactics: int,
        num_node_types: int = len(NODE_TYPE_TO_ID),
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_node_type: bool = True,
        max_args: int = 3,
    ) -> None:
        super().__init__()

        # Backbone — shared encoder + tactic head
        self.backbone = GraphSAGEStateClassifier(
            num_node_labels=num_node_labels,
            num_tactics=num_tactics,
            num_node_types=num_node_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_node_type=use_node_type,
        )

        # Tactic embedding (one learned vector per tactic family)
        self.tactic_embedding = nn.Embedding(num_tactics, hidden_dim)

        # Pointer head
        self.argument_selector = ArgumentSelector(hidden_dim)

        self.max_args = max_args
        self.hidden_dim = hidden_dim

    # ---- convenience accessors for the backbone ----
    @property
    def label_embedding(self) -> nn.Embedding:
        return self.backbone.label_embedding

    @property
    def node_type_embedding(self) -> nn.Embedding | None:
        return self.backbone.node_type_embedding

    def forward(
        self,
        data,
        *,
        teacher_tactic_ids: Tensor | None = None,
        tactic_names: list[str] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Return ``(tactic_logits, arg_logits_list)``.

        Parameters
        ----------
        data : PyG Batch
            Must include ``premise_mask`` (bool tensor, one per node) and
            ``batch`` (standard PyG batch vector).
        teacher_tactic_ids : Tensor, optional
            Ground-truth tactic ids for teacher-forcing during training.
            Shape ``[B]``.
        tactic_names : list[str], optional
            Tactic family names per sample (used to look up arity).
            If not provided the model uses ``max_args`` for every sample.

        Returns
        -------
        tactic_logits : Tensor, shape ``[B, num_tactics]``
        arg_logits_list : list[Tensor]
            One entry per argument step (up to ``max_args``).
            Each has shape ``[B, max_nodes_in_batch]``.
        """
        # 1. Encode all nodes
        node_embeddings = self.backbone.encode_nodes(data)  # [total_nodes, H]

        # 2. Readout the State node embedding per graph
        state_emb = self.backbone.readout(node_embeddings, data)  # [B, H]

        # 3. Tactic classification
        tactic_logits = self.backbone.classifier(
            self.backbone.dropout(state_emb)
        )  # [B, num_tactics]

        # 4. Determine which tactic embedding to use as query context
        if teacher_tactic_ids is not None:
            tactic_ids = teacher_tactic_ids
        else:
            tactic_ids = tactic_logits.argmax(dim=1)  # [B]

        tactic_emb = self.tactic_embedding(tactic_ids)  # [B, H]

        # 5. Determine how many argument steps to run
        if tactic_names is not None:
            n_steps = max(get_tactic_arity(name) for name in tactic_names)
            n_steps = min(n_steps, self.max_args)
        else:
            n_steps = self.max_args

        if n_steps == 0:
            return tactic_logits, []

        # Autoregressive argument selection
        # Overwrite the cache's premise_mask if it's too restrictive (e.g. missing 'app' or 'operator' nodes)
        # Type IDs: var=0, type=1, predicate=2, operator=3, app=4, meta=5
        # We allow selecting any of these common types as arguments.
        node_types = data.node_type.to(device=node_embeddings.device)
        premise_mask = (node_types >= 0) & (node_types <= 5)
        batch_index = data.batch.to(device=node_embeddings.device)

        arg_logits_list: list[Tensor] = []
        prev_arg_emb: Tensor | None = None

        for _ in range(n_steps):
            scores, selected_emb = self.argument_selector(
                state_emb, tactic_emb, node_embeddings, premise_mask, batch_index,
                prev_arg_emb=prev_arg_emb,
            )
            arg_logits_list.append(scores)
            prev_arg_emb = selected_emb

        return tactic_logits, arg_logits_list


# ---------------------------------------------------------------------------
# Combined loss computation
# ---------------------------------------------------------------------------


def resolve_arg_targets_to_padded(
    arg_node_indices: Tensor,
    batch_index: Tensor,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Remap global node indices to padded per-graph positions.

    Returns [B, max_gt_args] of positions into the padded [B, N_max] logit
    matrix, with -1 for invalid arguments.
    """
    offsets = torch.zeros_like(batch_index)
    for b in range(batch_size):
        graph_mask = batch_index == b
        offsets[graph_mask] = torch.arange(
            int(graph_mask.sum().item()), device=device, dtype=torch.long
        )

    result = arg_node_indices.clone().to(device)
    valid = result >= 0
    result[~valid] = -1

    flat_valid_indices = result[valid]
    total_nodes = batch_index.size(0)
    oob = (flat_valid_indices >= total_nodes)
    if oob.any():
        temp = result.clone()
        temp[valid] = torch.where(oob, torch.tensor(-1, device=device), offsets[flat_valid_indices.clamp(max=total_nodes - 1)])
        return temp

    result[valid] = offsets[flat_valid_indices]
    return result


def compute_combined_loss(
    tactic_logits: Tensor,
    arg_logits_list: list[Tensor],
    tactic_targets: Tensor,
    arg_targets: Tensor,
    batch_index: Tensor,
    *,
    tactic_arity_per_sample: list[int],
    arg_loss_weight: float = 0.5,
    unknown_tactic_id: int = 0,
    node_labels: Tensor | None = None,
    node_types: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Tactic classification loss + masked argument selection loss."""
    device = tactic_logits.device
    batch_size = tactic_logits.size(0)

    known_mask = tactic_targets != unknown_tactic_id
    if known_mask.any():
        tactic_loss = F.cross_entropy(tactic_logits[known_mask], tactic_targets[known_mask])
    else:
        tactic_loss = torch.tensor(0.0, device=device)

    if not arg_logits_list:
        return tactic_loss, {
            "tactic_loss": float(tactic_loss.item()),
            "arg_loss": 0.0,
            "total_loss": float(tactic_loss.item()),
        }

    padded_targets = resolve_arg_targets_to_padded(
        arg_targets, batch_index, batch_size, device
    )

    arg_losses: list[Tensor] = []
    for step_k, arg_logits_k in enumerate(arg_logits_list):
        if step_k >= padded_targets.size(1):
            break

        gt_k = padded_targets[:, step_k]  # [B]
        valid = gt_k >= 0
        for b_idx in range(batch_size):
            if tactic_arity_per_sample[b_idx] <= step_k:
                valid[b_idx] = False
            elif valid[b_idx]:
                # Skip target if it was masked out by premise_mask
                if torch.isneginf(arg_logits_k[b_idx, gt_k[b_idx]]):
                    valid[b_idx] = False

        if not valid.any():
            continue

        step_loss = F.cross_entropy(arg_logits_k[valid].clamp(min=-1e4), gt_k[valid])
        arg_losses.append(step_loss)

    if arg_losses:
        arg_loss = torch.stack(arg_losses).mean()
    else:
        arg_loss = torch.tensor(0.0, device=device)

    total_loss = tactic_loss + arg_loss_weight * arg_loss
    return total_loss, {
        "tactic_loss": float(tactic_loss.item()),
        "arg_loss": float(arg_loss.item()),
        "total_loss": float(total_loss.item()),
    }

