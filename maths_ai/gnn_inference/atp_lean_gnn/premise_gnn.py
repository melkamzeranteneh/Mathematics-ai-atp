"""Dedicated GNN encoder for premise selection.

Provides ``PremiseGNN``, a GATv2-based (or GraphSAGE-based) encoder that maps
proof-state graphs and library-lemma graphs into a shared embedding space.

    encode_nodes(graph_batch) -> Tensor  [num_nodes, hidden_dim]
    readout(node_embeddings, graph_batch) -> Tensor  [batch_size, hidden_dim]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, SAGEConv, global_mean_pool

from .pyg import NODE_TYPE_TO_ID


class PremiseGNN(nn.Module):
    """GATv2 (or GraphSAGE) encoder for proof-state and lemma graphs.

    Parameters
    ----------
    num_node_labels : int
        Vocabulary size for node label embeddings.
    num_tactics : int
        Vocabulary size for tactic embeddings.
    hidden_dim : int
        Width of all hidden layers and the output embedding.
    num_layers : int
        Number of message-passing layers (default 3).
    heads : int
        Number of attention heads per GATv2 layer (ignored for SAGE).
    dropout : float
        Dropout applied between layers.
    backbone : str
        ``"gatv2"`` (default) or ``"sage"`` for the ablation in §7.
    num_node_types : int
        Size of the node-type embedding table.
    """

    def __init__(
        self,
        num_node_labels: int,
        num_tactics: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        backbone: str = "gatv2",
        num_node_types: int = len(NODE_TYPE_TO_ID),
    ) -> None:
        super().__init__()

        if backbone not in {"gatv2", "sage"}:
            raise ValueError(f"backbone must be 'gatv2' or 'sage', got {backbone!r}")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        
        if backbone == "gatv2" and hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}) for GATv2"
            )

        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.num_layers = num_layers

        # Node feature embeddings
        self.label_embedding = nn.Embedding(num_node_labels, hidden_dim)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.is_bound_embedding = nn.Embedding(2, hidden_dim)
        self.binder_depth_embedding = nn.Embedding(10, hidden_dim)
        self.binder_kind_embedding = nn.Embedding(6, hidden_dim)

        self.tactic_embedding = nn.Embedding(num_tactics, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj_ar = nn.Linear(hidden_dim * 3, hidden_dim)

        # Message-passing layers
        self.convs = nn.ModuleList()
        if backbone == "gatv2":
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            )
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
                )
        else:  # sage
            for _ in range(num_layers):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def encode_nodes(self, graph_batch) -> Tensor:
        """Embed every node in the batched graph."""
        x = self.label_embedding(graph_batch.x)
        x = x + self.node_type_embedding(graph_batch.node_type)

        if hasattr(graph_batch, "is_bound"):
            x = x + self.is_bound_embedding(graph_batch.is_bound)
        if hasattr(graph_batch, "binder_depth"):
            x = x + self.binder_depth_embedding(graph_batch.binder_depth)
        if hasattr(graph_batch, "binder_kind"):
            x = x + self.binder_kind_embedding(graph_batch.binder_kind)

        for i, conv in enumerate(self.convs):
            x = conv(x, graph_batch.edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)

        return x

    def readout(self, node_embeddings: Tensor, graph_batch) -> Tensor:
        """Read out one embedding per graph using the State root node.
        
        Falls back to global mean pooling when state_node_index is absent.
        """
        if hasattr(graph_batch, "state_node_index"):
            idx = graph_batch.state_node_index
            if not torch.is_tensor(idx):
                idx = torch.tensor([int(idx)], dtype=torch.long, device=node_embeddings.device)
            else:
                idx = idx.to(device=node_embeddings.device, dtype=torch.long).view(-1)
            
            # Safety clamp
            idx = idx.clamp(0, node_embeddings.size(0) - 1)
            return node_embeddings.index_select(0, idx)

        return global_mean_pool(node_embeddings, graph_batch.batch)

    def forward(self, graph_batch) -> Tensor:
        """Convenience wrapper: encode_nodes then readout."""
        node_embeddings = self.encode_nodes(graph_batch)
        return self.readout(node_embeddings, graph_batch)

    def score_candidates(
        self,
        state_vecs: Tensor,
        tactic_embs: Tensor,
        pools,
        temperature: float = 0.07,
    ) -> list[Tensor]:
        """Score all candidates in each pool."""
        from .premise_pool import CandidatePool
        
        batch_size = state_vecs.size(0)
        if len(pools) != batch_size:
            raise ValueError(
                f"Number of pools ({len(pools)}) does not match "
                f"batch size ({batch_size})."
            )

        score_list: list[Tensor] = []
        for b in range(batch_size):
            query = self.query_proj(
                torch.cat([state_vecs[b], tactic_embs[b]], dim=-1)
            )
            candidate_keys = self.key_proj(pools[b].candidate_vectors)
            scores = (candidate_keys @ query) / temperature
            score_list.append(scores)

        return score_list

    def score_and_select_arguments(
        self,
        state_vec: Tensor,
        tactic_emb: Tensor,
        candidate_vectors: Tensor,
        candidate_sources: list[str],
        candidate_ids: list[int],
        num_args: int,
        temperature: float = 0.07,
    ) -> tuple[list[int], list[float], list[str], list[int]]:
        """Autoregressively select arguments from the pool."""
        num_args = min(num_args, candidate_vectors.size(0))
        if num_args <= 0:
            return [], [], [], []

        state = state_vec.view(-1)
        tactic = tactic_emb.view(-1)
        keys = self.key_proj(candidate_vectors)

        selected_indices: list[int] = []
        selected_scores: list[float] = []
        selected_sources: list[str] = []
        selected_ids: list[int] = []
        
        mask = torch.zeros(candidate_vectors.size(0), dtype=torch.bool, device=state.device)
        query = self.query_proj(torch.cat([state, tactic], dim=0))

        for _ in range(num_args):
            raw_scores = (keys @ query) / temperature
            raw_scores = raw_scores.masked_fill(mask, float("-inf"))
            
            best = int(raw_scores.argmax().item())
            selected_indices.append(best)
            selected_scores.append(float(raw_scores[best].item()))
            selected_sources.append(candidate_sources[best])
            selected_ids.append(candidate_ids[best])
            
            mask[best] = True
            prev = candidate_vectors[best].view(-1)
            query = self.query_proj_ar(torch.cat([state, tactic, prev], dim=0))

        return selected_indices, selected_scores, selected_sources, selected_ids

    def _encode_cached(
        self,
        graph_batch,
        cache: dict[int, Tensor],
        graph_ids: list[int],
    ) -> Tensor:
        """Encode graphs, reusing cached embeddings for ids already seen.

        ``graph_batch`` must hold exactly the graphs whose ids miss the cache,
        in the same order those ids appear in ``graph_ids``.
        """
        device = next(self.parameters()).device
        results: dict[int, Tensor] = {}

        miss_positions: list[int] = []
        for pos, gid in enumerate(graph_ids):
            cached = cache.get(gid)
            if cached is None:
                miss_positions.append(pos)
                continue
            cached = cached.to(device)
            cache[gid] = cached
            results[pos] = cached

        if miss_positions:
            if graph_batch is None:
                raise ValueError("Cache miss but graph_batch is None.")
            num_graphs = int(getattr(graph_batch, "num_graphs", 1))
            if num_graphs != len(miss_positions):
                raise ValueError(
                    f"graph_batch contains {num_graphs} graphs "
                    f"but there are {len(miss_positions)} cache misses."
                )
            with torch.no_grad():
                node_embs = self.encode_nodes(graph_batch)
                state_embs = self.readout(node_embs, graph_batch)
            for i, pos in enumerate(miss_positions):
                # clone: a row of state_embs is a view that would otherwise
                # keep the whole batch's storage alive for as long as it is cached
                vec = state_embs[i].clone()
                cache[graph_ids[pos]] = vec
                results[pos] = vec

        return torch.stack([results[pos] for pos in range(len(graph_ids))], dim=0)

    def encode_lemma_cached(
        self,
        graph_batch,
        cache: dict[int, Tensor],
        lemma_ids: list[int],
    ) -> Tensor:
        """Encode lemma graphs, reusing cached embeddings."""
        return self._encode_cached(graph_batch, cache, lemma_ids)

    def encode_state_cached(
        self,
        graph_batch,
        cache: dict[int, Tensor],
        state_ids: list[int],
    ) -> Tensor:
        """Encode state graphs, reusing cached embeddings."""
        return self._encode_cached(graph_batch, cache, state_ids)

    def get_state_and_local_embeddings(
        self,
        graph_batch,
    ) -> tuple[Tensor, Tensor]:
        """Get both state and node embeddings in one forward pass."""
        node_embeddings = self.encode_nodes(graph_batch)
        state_embeddings = self.readout(node_embeddings, graph_batch)
        return state_embeddings, node_embeddings