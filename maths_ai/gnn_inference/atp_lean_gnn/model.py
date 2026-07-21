from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, SAGEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as graph_softmax

from .pyg import NODE_TYPE_TO_ID


VALID_READOUTS = ("state", "state_mean_attention")


def _state_node_embeddings(node_embeddings: torch.Tensor, data) -> torch.Tensor:
    if not hasattr(data, "state_node_index"):
        raise ValueError("Batched graph data is missing the 'state_node_index' attribute.")

    state_node_index = data.state_node_index
    if not torch.is_tensor(state_node_index):
        state_node_index = torch.tensor(
            [int(state_node_index)],
            device=node_embeddings.device,
            dtype=torch.long,
        )
    else:
        state_node_index = state_node_index.to(
            device=node_embeddings.device,
            dtype=torch.long,
        ).view(-1)
    return node_embeddings.index_select(0, state_node_index)


class StateMeanAttentionReadout(nn.Module):
    """Fuse state, global mean, and state-conditioned attention summaries."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        state_embeddings: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_index = batch_index.to(device=node_embeddings.device, dtype=torch.long)
        graph_count = state_embeddings.size(0)
        state_per_node = state_embeddings.index_select(0, batch_index)
        scores = self.attention_score(
            torch.tanh(
                self.node_projection(node_embeddings)
                + self.state_projection(state_per_node)
            )
        ).squeeze(-1)
        attention_weights = graph_softmax(
            scores,
            batch_index,
            num_nodes=graph_count,
        )
        weighted_embeddings = global_add_pool(
            attention_weights.unsqueeze(-1) * node_embeddings,
            batch_index,
            size=graph_count,
        )
        mean_embeddings = global_mean_pool(
            node_embeddings,
            batch_index,
            size=graph_count,
        )
        combined = torch.cat(
            [state_embeddings, mean_embeddings, weighted_embeddings],
            dim=-1,
        )
        fused = self.fusion(combined)
        return F.gelu(self.normalization(fused)), attention_weights


@dataclass(frozen=True)
class GraphSAGEClassifierConfig:
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


@dataclass(frozen=True)
class GATv2ClassifierConfig:
    """Configuration for :class:`GATv2StateClassifier`.

    ``hidden_dim`` is the total width across all attention heads.  Each head
    therefore emits ``hidden_dim // heads`` features and concatenation restores
    the configured width.  No edge attributes are required; the underlying
    ``GATv2Conv`` simply attends over ``edge_index``.
    """

    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.2
    heads: int = 8
    readout: str = "state"

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "heads": self.heads,
            "readout": self.readout,
        }


class GraphSAGEStateClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_node_labels: int,
        num_tactics: int,
        num_node_types: int = len(NODE_TYPE_TO_ID),
        num_binder_kinds: int = 6,
        max_binder_depth: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_node_type: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("GraphSAGEStateClassifier requires at least one message-passing layer.")

        self.label_embedding = nn.Embedding(num_node_labels, hidden_dim)
        self.node_type_embedding = (
            nn.Embedding(num_node_types, hidden_dim) if use_node_type else None
        )

        # Binder feature embeddings
        self.is_bound_embedding = nn.Embedding(2, hidden_dim)  # 0/1
        self.binder_depth_embedding = nn.Embedding(max_binder_depth, hidden_dim)
        self.binder_kind_embedding = nn.Embedding(num_binder_kinds, hidden_dim)

        self.convs = nn.ModuleList(
            SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_tactics)

    def encode_nodes(self, data) -> torch.Tensor:
        x = self.label_embedding(data.x)
        if self.node_type_embedding is not None:
            x = x + self.node_type_embedding(data.node_type)

        # Add binder features if present
        if hasattr(data, "is_bound"):
            x = x + self.is_bound_embedding(data.is_bound)
        if hasattr(data, "binder_depth"):
            x = x + self.binder_depth_embedding(data.binder_depth)
        if hasattr(data, "binder_kind"):
            x = x + self.binder_kind_embedding(data.binder_kind)

        for index, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            x = F.relu(x)
            if index < len(self.convs) - 1:
                x = self.dropout(x)
        return x

    def readout(self, node_embeddings: torch.Tensor, data) -> torch.Tensor:
        if not hasattr(data, "state_node_index"):
            raise ValueError("Batched graph data is missing the 'state_node_index' attribute.")

        state_node_index = data.state_node_index
        if not torch.is_tensor(state_node_index):
            state_node_index = torch.tensor(
                [int(state_node_index)],
                device=node_embeddings.device,
                dtype=torch.long,
            )
        else:
            state_node_index = state_node_index.to(
                device=node_embeddings.device,
                dtype=torch.long,
            ).view(-1)

        return node_embeddings.index_select(0, state_node_index)

    def forward(self, data) -> torch.Tensor:
        node_embeddings = self.encode_nodes(data)
        graph_embeddings = self.readout(node_embeddings, data)
        return self.classifier(self.dropout(graph_embeddings))


class GATv2StateClassifier(nn.Module):
    """Graph attention (GATv2) drop-in replacement for :class:`GraphSAGEStateClassifier`.

    Uses :class:`~torch_geometric.nn.GATv2Conv`, which is strictly at least as
    expressive as the original GAT and additionally supports edge attributes via
    ``edge_dim`` for future typed-edge extensions. Each head emits a share of the
    configured width, so concatenating all heads restores ``hidden_dim`` while the
    classifier head and downstream accessors remain GraphSAGE-compatible.
    """

    def __init__(
        self,
        *,
        num_node_labels: int,
        num_tactics: int,
        num_node_types: int = len(NODE_TYPE_TO_ID),
        num_binder_kinds: int = 6,
        max_binder_depth: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.2,
        heads: int = 8,
        use_node_type: bool = True,
        readout: str = "state",
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("GATv2StateClassifier requires at least one message-passing layer.")
        if heads < 1:
            raise ValueError("GATv2StateClassifier requires at least one attention head.")
        if hidden_dim % heads != 0:
            raise ValueError(
                "GATv2StateClassifier requires hidden_dim to be divisible by heads "
                f"(got hidden_dim={hidden_dim}, heads={heads})."
            )
        normalized_readout = readout.lower().strip()
        if normalized_readout not in VALID_READOUTS:
            raise ValueError(
                f"GATv2StateClassifier readout must be one of: {', '.join(VALID_READOUTS)}."
            )

        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.readout_mode = normalized_readout
        self.label_embedding = nn.Embedding(num_node_labels, hidden_dim)
        self.node_type_embedding = (
            nn.Embedding(num_node_types, hidden_dim) if use_node_type else None
        )

        self.is_bound_embedding = nn.Embedding(2, hidden_dim)
        self.binder_depth_embedding = nn.Embedding(max_binder_depth, hidden_dim)
        self.binder_kind_embedding = nn.Embedding(num_binder_kinds, hidden_dim)

        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden_dim,
                self.head_dim,
                heads=heads,
                dropout=dropout,
                concat=True,
            )
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.global_readout = (
            StateMeanAttentionReadout(hidden_dim)
            if self.readout_mode == "state_mean_attention"
            else None
        )
        self.classifier = nn.Linear(hidden_dim, num_tactics)

    def encode_nodes(self, data) -> torch.Tensor:
        x = self.label_embedding(data.x)
        if self.node_type_embedding is not None:
            x = x + self.node_type_embedding(data.node_type)

        if hasattr(data, "is_bound"):
            x = x + self.is_bound_embedding(data.is_bound)
        if hasattr(data, "binder_depth"):
            x = x + self.binder_depth_embedding(data.binder_depth)
        if hasattr(data, "binder_kind"):
            x = x + self.binder_kind_embedding(data.binder_kind)

        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return x

    def readout_with_details(
        self,
        node_embeddings: torch.Tensor,
        data,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state_embeddings = _state_node_embeddings(node_embeddings, data)
        if self.global_readout is None:
            return state_embeddings, {}
        batch_index = getattr(data, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(
                node_embeddings.size(0),
                device=node_embeddings.device,
                dtype=torch.long,
            )
        graph_embeddings, attention_weights = self.global_readout(
            node_embeddings,
            state_embeddings,
            batch_index,
        )
        return graph_embeddings, {"attention_weights": attention_weights}

    def readout(self, node_embeddings: torch.Tensor, data) -> torch.Tensor:
        graph_embeddings, _ = self.readout_with_details(node_embeddings, data)
        return graph_embeddings

    def forward_with_readout_details(
        self,
        data,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return logits plus optional global-readout diagnostics.

        The normal ``forward`` path delegates here, so analysis can inspect the
        learned pooling weights without registering hooks or changing numerical
        behavior during training.
        """
        node_embeddings = self.encode_nodes(data)
        graph_embeddings, details = self.readout_with_details(node_embeddings, data)
        return self.classifier(self.dropout(graph_embeddings)), details

    def forward(self, data) -> torch.Tensor:
        logits, _ = self.forward_with_readout_details(data)
        return logits
