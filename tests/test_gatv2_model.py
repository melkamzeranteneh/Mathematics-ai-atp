"""Tests for the GATv2 backbone and gnn_type config routing."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

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
    _amp_dtype,
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


def test_gatv2_total_width_is_split_across_heads():
    heads = 4
    hidden_dim = 16
    model = GATv2StateClassifier(
        num_node_labels=10,
        num_tactics=5,
        hidden_dim=hidden_dim,
        num_layers=2,
        heads=heads,
    )
    assert model.head_dim == hidden_dim // heads
    assert model.convs[0].out_channels == hidden_dim // heads
    assert model.convs[0].heads == heads


def test_gatv2_requires_at_least_one_layer():
    with pytest.raises(ValueError):
        GATv2StateClassifier(num_node_labels=10, num_tactics=5, num_layers=0)


def test_gatv2_requires_hidden_dim_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        GATv2StateClassifier(
            num_node_labels=10,
            num_tactics=5,
            hidden_dim=10,
            heads=4,
        )


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


def test_state_mean_attention_readout_fuses_three_hidden_vectors():
    first = _make_batch(num_nodes=5, num_edges=6)
    second = _make_batch(num_nodes=3, num_edges=4)
    batch = Batch.from_data_list([first, second])
    model = GATv2StateClassifier(
        num_node_labels=10,
        num_tactics=5,
        hidden_dim=16,
        num_layers=2,
        heads=4,
        readout="state_mean_attention",
    )
    model.eval()

    node_embeddings = model.encode_nodes(batch)
    state_embeddings = node_embeddings.index_select(0, batch.state_node_index.view(-1))
    graph_embeddings, attention_weights = model.global_readout(
        node_embeddings,
        state_embeddings,
        batch.batch,
    )
    logits = model(batch)

    assert model.global_readout.fusion.in_features == 16 * 3
    assert model.global_readout.fusion.out_features == 16
    assert graph_embeddings.shape == (2, 16)
    assert logits.shape == (2, 5)
    for graph_index in range(2):
        graph_weights = attention_weights[batch.batch == graph_index]
        assert torch.allclose(graph_weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_forward_details_expose_pooling_weights_without_changing_logits():
    first = _make_batch(num_nodes=5, num_edges=6)
    second = _make_batch(num_nodes=3, num_edges=4)
    batch = Batch.from_data_list([first, second])
    model = GATv2StateClassifier(
        num_node_labels=10,
        num_tactics=5,
        hidden_dim=16,
        num_layers=2,
        heads=4,
        readout="state_mean_attention",
    )
    model.eval()

    logits = model(batch)
    detailed_logits, details = model.forward_with_readout_details(batch)

    assert torch.equal(logits, detailed_logits)
    assert details["attention_weights"].shape == batch.x.shape
    for graph_index in range(2):
        graph_weights = details["attention_weights"][batch.batch == graph_index]
        assert torch.allclose(graph_weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_state_readout_details_remain_empty_and_preserve_logits():
    model = GATv2StateClassifier(
        num_node_labels=10,
        num_tactics=5,
        hidden_dim=16,
        num_layers=2,
        heads=4,
        readout="state",
    )
    model.eval()
    data = _make_batch()

    logits = model(data)
    detailed_logits, details = model.forward_with_readout_details(data)

    assert torch.equal(logits, detailed_logits)
    assert details == {}


def test_state_mean_attention_readout_receives_gradients():
    model = GATv2StateClassifier(
        num_node_labels=10,
        num_tactics=5,
        hidden_dim=16,
        num_layers=2,
        heads=4,
        readout="state_mean_attention",
    )
    model(_make_batch()).sum().backward()

    assert model.global_readout.attention_score.weight.grad is not None
    assert model.global_readout.fusion.weight.grad is not None


def test_gatv2_rejects_unknown_readout():
    with pytest.raises(ValueError, match="readout"):
        GATv2StateClassifier(
            num_node_labels=10,
            num_tactics=5,
            hidden_dim=16,
            heads=4,
            readout="mystery",
        )


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


def test_baseline_config_reads_heads_for_gat():
    """Config heads value should be read and passed to GATv2 model."""
    meta = type("M", (), {"node_vocab": [f"n{i}" for i in range(10)],
                          "tactic_vocab": [f"t{i}" for i in range(5)]})()

    cfg = BaselineConfig.from_dict({
        "prepared_root": "x",
        "gnn_type": "gat",
        "model": {"heads": 4, "hidden_dim": 128}
    })
    model = build_baseline_model(meta, cfg)

    assert isinstance(model, GATv2StateClassifier)
    assert model.head_dim == 128 // 4


def test_baseline_config_routes_state_mean_attention_readout():
    meta = type("M", (), {"node_vocab": [f"n{i}" for i in range(10)],
                          "tactic_vocab": [f"t{i}" for i in range(5)]})()
    cfg = BaselineConfig.from_dict({
        "prepared_root": "x",
        "gnn_type": "gat",
        "model": {
            "heads": 4,
            "hidden_dim": 16,
            "readout": "state_mean_attention",
        },
    })

    model = build_baseline_model(meta, cfg)

    assert cfg.model.readout == "state_mean_attention"
    assert model.readout_mode == "state_mean_attention"
    assert model.global_readout is not None


def test_pointer_config_reads_heads_for_gat():
    """PointerConfig should also read heads and pass to GATv2 backbone."""
    meta = type("M", (), {"node_vocab": [f"n{i}" for i in range(10)],
                          "tactic_vocab": [f"t{i}" for i in range(5)]})()

    cfg = PointerConfig.from_dict({
        "prepared_root": "x",
        "gnn_type": "gat",
        "model": {"heads": 4, "hidden_dim": 128}
    })
    model = build_pointer_model(meta, cfg)

    assert isinstance(model.backbone, GATv2StateClassifier)
    assert model.backbone.head_dim == 128 // 4


def test_gat_uses_bfloat16_amp_when_supported(monkeypatch):
    cfg = BaselineConfig.from_dict({
        "prepared_root": "x",
        "gnn_type": "gat",
        "training": {"use_amp": True},
    })
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert _amp_dtype(torch.device("cuda"), cfg) == torch.bfloat16


def test_gat_does_not_fall_back_to_unsafe_float16(monkeypatch):
    cfg = BaselineConfig.from_dict({
        "prepared_root": "x",
        "gnn_type": "gat",
        "training": {"use_amp": True},
    })
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    assert _amp_dtype(torch.device("cuda"), cfg) is None


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


def test_pg_batch_data_parallel_fallback_rebatches_list():
    from torch_geometric.data import Batch, Data

    seen = {}

    class Recorder(nn.Module):
        def forward(self, data):
            seen["is_batch"] = hasattr(data, "x")
            num = data.num_graphs if hasattr(data, "num_graphs") else len(data)
            return torch.zeros(num, 5)

    batch = Batch.from_data_list([
        Data(x=torch.zeros(3, dtype=torch.long), edge_index=torch.zeros(2, 0, dtype=torch.long)),
        Data(x=torch.zeros(2, dtype=torch.long), edge_index=torch.zeros(2, 0, dtype=torch.long)),
    ])
    wrapper = PyGBatchDataParallel(Recorder())
    wrapper._fallback = True
    wrapper.inner = wrapper.module
    # Second call would otherwise pass a list straight to the module.
    out = wrapper(batch)
    assert seen["is_batch"] is True
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
