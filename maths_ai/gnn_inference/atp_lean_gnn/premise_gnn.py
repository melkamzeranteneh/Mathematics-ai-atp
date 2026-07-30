"""Dedicated GNN encoder for premise selection.

Provides ``PremiseGNN``, a GATv2-based (or GraphSAGE-based) encoder that maps
proof-state graphs and library-lemma graphs into a shared embedding space.

Contract (both Person A and Person B depend on this):
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

        self.hidden_dim = hidden_dim
        self.backbone = backbone

        # Node feature embeddings (same scheme as GraphSAGEStateClassifier)
        self.label_embedding = nn.Embedding(num_node_labels, hidden_dim)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.is_bound_embedding = nn.Embedding(2, hidden_dim)
        self.binder_depth_embedding = nn.Embedding(10, hidden_dim)
        self.binder_kind_embedding = nn.Embedding(6, hidden_dim)

        # Message-passing layers
        self.convs = nn.ModuleList()
        if backbone == "gatv2":
            # First layer: hidden_dim -> hidden_dim (multi-head, then concat)
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

    # ------------------------------------------------------------------
    # Contracted interface
    # ------------------------------------------------------------------

    def encode_nodes(self, graph_batch) -> Tensor:
        """Embed every node in the batched graph.

        Parameters
        ----------
        graph_batch : torch_geometric.data.Batch
            Batched PyG graph with attributes ``x``, ``edge_index``,
            ``node_type``, and optionally ``is_bound``, ``binder_depth``,
            ``binder_kind``.

        Returns
        -------
        Tensor
            Node embeddings, shape ``[num_nodes, hidden_dim]``.
        """
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

        return x  # [num_nodes, hidden_dim]

    def readout(self, node_embeddings: Tensor, graph_batch) -> Tensor:
        """Read out one embedding per graph using the State root node.

        Falls back to global mean pooling when ``state_node_index`` is absent
        (e.g. when encoding a library lemma graph that has no State node).

        Parameters
        ----------
        node_embeddings : Tensor
            Output of ``encode_nodes``, shape ``[num_nodes, hidden_dim]``.
        graph_batch : torch_geometric.data.Batch

        Returns
        -------
        Tensor
            Graph-level embeddings, shape ``[batch_size, hidden_dim]``.
        """
        if hasattr(graph_batch, "state_node_index"):
            idx = graph_batch.state_node_index
            if not torch.is_tensor(idx):
                idx = torch.tensor([int(idx)], dtype=torch.long, device=node_embeddings.device)
            else:
                idx = idx.to(device=node_embeddings.device, dtype=torch.long).view(-1)
            # idx contains per-graph local offsets; add the per-graph node offset
            # so they become global indices into the batched node_embeddings tensor.
            if hasattr(graph_batch, "ptr"):
                ptr = graph_batch.ptr.to(device=node_embeddings.device)
                # ptr[i] is the start of graph i in the batched node list
                idx = idx + ptr[:-1]
            return node_embeddings.index_select(0, idx)  # [batch_size, hidden_dim]

        # Lemma graphs: no State node — use mean pooling
        return global_mean_pool(node_embeddings, graph_batch.batch)  # [batch_size, hidden_dim]

    def forward(self, graph_batch) -> Tensor:
        """Convenience wrapper: encode_nodes then readout."""
        node_embeddings = self.encode_nodes(graph_batch)
        return self.readout(node_embeddings, graph_batch)

    # ------------------------------------------------------------------
    # Per-node embedding cache (inference search loop)
    # ------------------------------------------------------------------

    def encode_lemma_cached(
        self,
        graph_batch,
        cache: dict[int, Tensor],
        lemma_ids: list[int],
    ) -> Tensor:
        """Encode a batch of lemma graphs, reusing cached embeddings.

        During the proof-search loop the same library lemma may be scored
        against many different proof states.  This method avoids re-running
        the GNN for lemmas whose embedding is already in ``cache``.

        Parameters
        ----------
        graph_batch : torch_geometric.data.Batch | None
            Batched lemma graphs for the cache-miss lemmas only, in the same
            order as the miss positions.  Pass ``None`` when all lemmas are
            already cached.
        cache : dict[int, Tensor]
            Maps lemma_id -> embedding vector ``[hidden_dim]``.  Updated
            in-place with newly computed embeddings.
        lemma_ids : list[int]
            Lemma id for each requested lemma, in the desired output order.

        Returns
        -------
        Tensor
            Embeddings for all requested lemmas (cached + freshly computed),
            shape ``[len(lemma_ids), hidden_dim]``, in the original order.
        """
        device = next(self.parameters()).device
        results: dict[int, Tensor] = {}

        # Separate hits from misses
        miss_positions: list[int] = []
        for pos, lid in enumerate(lemma_ids):
            if lid in cache:
                results[pos] = cache[lid]
            else:
                miss_positions.append(pos)

        # Compute only the misses
        if miss_positions:
            if graph_batch is None:
                raise ValueError(
                    f"{len(miss_positions)} lemma(s) not in cache but graph_batch is None."
                )
            if len(miss_positions) != graph_batch.num_graphs:
                raise ValueError(
                    f"graph_batch contains {graph_batch.num_graphs} graphs "
                    f"but there are {len(miss_positions)} cache misses. "
                    "Pass only the miss graphs in graph_batch."
                )
            with torch.no_grad():
                node_embs = self.encode_nodes(graph_batch)
                state_embs = self.readout(node_embs, graph_batch)  # [num_misses, H]
            for i, pos in enumerate(miss_positions):
                lid = lemma_ids[pos]
                vec = state_embs[i].to(device)
                cache[lid] = vec
                results[pos] = vec

        return torch.stack([results[pos] for pos in range(len(lemma_ids))], dim=0)
