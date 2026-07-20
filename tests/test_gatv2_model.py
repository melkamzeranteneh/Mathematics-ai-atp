"""Tests for the GATv2 backbone and gnn_type config routing."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from maths_ai.gnn_inference.atp_lean_gnn.model import (
    GATv2StateClassifier,
    GraphSAGEStateClassifier,
)
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    BaselineConfig,
    PointerConfig,
    build_baseline_model,
    build_pointer_model,
)
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier


def _make_batch(num_nodes: int = 6, num_edges: int = 8, hidden_dim: int = 16, heads: int = 4):
    torch.manual_seed(0)
    x = torch.randint(0, 10, (num_nodes,))
    node_type = torch.zeros(num_nodes, dtype=torch.long)
    edge_pairs = torch.randint(0, num_nodes, (num_edges, 2)).t().contiguous()
    state_node_index = torch.tensor([0], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    data = Data(x=x, node_type=node_type, edge_index=edge_pairs,
                state_node_index=state_node_index, batch=batch)
    return data


def test_gatv2_forward_shape():
    model = GATv2StateClassifier(
        num_node_labels=10,
        num_tactics=5,
        hidden_dim=16,
        num_layers=3,
        heads=4,
    )
    data = _make_batch(hidden_dim=16, heads=4)
    out = model(data)
    assert out.shape == (1, 5)


def test_gatv2_head_proj_width():
    heads = 4
    hidden_dim = 16
    model = GATv2StateClassifier(
        num_node_labels=10,
        num_tactics=5,
        hidden_dim=hidden_dim,
        num_layers=2,
        heads=heads,
    )
    assert model.head_proj.in_features == heads * hidden_dim
    assert model.head_proj.out_features == hidden_dim


def test_gatv2_requires_at_least_one_layer():
    with pytest.raises(ValueError):
        GATv2StateClassifier(num_node_labels=10, num_tactics=5, num_layers=0)


def test_gatv2_does_not_require_edge_attr():
    model = GATv2StateClassifier(num_node_labels=10, num_tactics=5, hidden_dim=16, heads=2)
    data = _make_batch()
    out = model(data)
    assert out.shape[0] == 1


def test_gatv2_and_sage_shapes_match():
    sage = GraphSAGEStateClassifier(num_node_labels=10, num_tactics=5, hidden_dim=16, num_layers=3)
    gat = GATv2StateClassifier(num_node_labels=10, num_tactics=5, hidden_dim=16, num_layers=3, heads=4)
    data = _make_batch()
    assert sage(data).shape == gat(data).shape


def test_baseline_build_routes_gnn_type():
    meta = type("M", (), {"node_vocab": [f"n{i}" for i in range(10)],
                          "tactic_vocab": [f"t{i}" for i in range(5)]})()

    sage_cfg = BaselineConfig.from_dict({"prepared_root": "x", "gnn_type": "sage"})
    gat_cfg = BaselineConfig.from_dict({"prepared_root": "x", "gnn_type": "gat"})

    sage_model = build_baseline_model(meta, sage_cfg)
    gat_model = build_baseline_model(meta, gat_cfg)

    assert isinstance(sage_model, GraphSAGEStateClassifier)
    assert isinstance(gat_model, GATv2StateClassifier)


def test_pointer_build_routes_gnn_type():
    meta = type("M", (), {"node_vocab": [f"n{i}" for i in range(10)],
                          "tactic_vocab": [f"t{i}" for i in range(5)]})()

    sage_cfg = PointerConfig.from_dict({"prepared_root": "x", "gnn_type": "sage"})
    gat_cfg = PointerConfig.from_dict({"prepared_root": "x", "gnn_type": "gat"})

    sage_model = build_pointer_model(meta, sage_cfg)
    gat_model = build_pointer_model(meta, gat_cfg)

    assert isinstance(sage_model.backbone, GraphSAGEStateClassifier)
    assert isinstance(gat_model.backbone, GATv2StateClassifier)


def test_tactic_with_args_backbone_switch():
    sage = TacticWithArgsClassifier(num_node_labels=10, num_tactics=5, gnn_type="sage")
    gat = TacticWithArgsClassifier(num_node_labels=10, num_tactics=5, gnn_type="gat")
    assert isinstance(sage.backbone, GraphSAGEStateClassifier)
    assert isinstance(gat.backbone, GATv2StateClassifier)


def test_baseline_config_rejects_bad_gnn_type():
    with pytest.raises(ValueError):
        BaselineConfig.from_dict({"prepared_root": "x", "gnn_type": "transformer"})


def test_pointer_config_rejects_bad_gnn_type():
    with pytest.raises(ValueError):
        PointerConfig.from_dict({"prepared_root": "x", "gnn_type": "transformer"})
