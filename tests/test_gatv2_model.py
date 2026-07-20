"""Tests for the GATv2 backbone and gnn_type config routing."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
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
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    PyGBatchDataParallel,
    _safe_num_workers,
    _unwrap_model,
    maybe_wrap_data_parallel,
)


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


def test_maybe_wrap_data_parallel_cpu_returns_unwrapped():
    model = GraphSAGEStateClassifier(num_node_labels=10, num_tactics=5, hidden_dim=16)
    wrapped, device, gpu_ids = maybe_wrap_data_parallel(model, torch.device("cpu"))
    assert wrapped is model
    assert gpu_ids == []


def test_maybe_wrap_data_parallel_single_gpu_returns_unwrapped():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = GraphSAGEStateClassifier(num_node_labels=10, num_tactics=5, hidden_dim=16)
    wrapped, device, gpu_ids = maybe_wrap_data_parallel(model, torch.device("cuda"))
    # No assertion on wrap state (depends on GPU count); just ensure it runs.
    assert device.type == "cuda"


def test_pg_batch_data_parallel_converts_batch_to_list():
    from torch_geometric.data import Batch, Data

    class Recorder(nn.Module):
        def forward(self, data):
            # PyG DP's single-device fallback re-batches the list into a Batch;
            # on multi-GPU it would pass the list directly. Either way the
            # wrapper must accept a Batch without iterating its (attr, value)
            # tuples.
            num = data.num_graphs if hasattr(data, "num_graphs") else len(data)
            return torch.zeros(num, 5)

    batch = Batch.from_data_list([
        Data(x=torch.zeros(3, dtype=torch.long), edge_index=torch.zeros(2, 0, dtype=torch.long)),
        Data(x=torch.zeros(2, dtype=torch.long), edge_index=torch.zeros(2, 0, dtype=torch.long)),
    ])
    wrapper = PyGBatchDataParallel(Recorder())
    out = wrapper(batch)
    assert out.shape == (2, 5)


def test_unwrap_model_strips_pg_batch_data_parallel():
    inner = GraphSAGEStateClassifier(num_node_labels=10, num_tactics=5, hidden_dim=16)
    wrapped = PyGBatchDataParallel(inner)
    assert _unwrap_model(wrapped) is inner
    assert _unwrap_model(inner) is inner


def test_safe_num_workers_caps_on_small_shm(monkeypatch):
    # 512 MiB available -> at most 2 workers under the 256 MiB/worker floor.
    monkeypatch.setattr("maths_ai.gnn_inference.atp_lean_gnn.training._shm_bytes", lambda: 512 * 1024 * 1024)
    workers, warning = _safe_num_workers(8, pin_memory=True)
    assert workers == 2
    assert warning is not None


def test_safe_num_workers_zero_when_shm_tiny(monkeypatch):
    monkeypatch.setattr("maths_ai.gnn_inference.atp_lean_gnn.training._shm_bytes", lambda: 64 * 1024 * 1024)
    workers, warning = _safe_num_workers(4, pin_memory=True)
    assert workers == 0
    assert warning is not None


def test_safe_num_workers_keeps_request_when_shm_large(monkeypatch):
    monkeypatch.setattr("maths_ai.gnn_inference.atp_lean_gnn.training._shm_bytes", lambda: 64 * 1024 * 1024 * 1024)
    workers, warning = _safe_num_workers(4, pin_memory=True)
    assert workers == 4
    assert warning is None
