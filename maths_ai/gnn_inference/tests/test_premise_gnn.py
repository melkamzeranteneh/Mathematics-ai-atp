"""Tests for PremiseGNN — encode_nodes / readout / select_arguments contract."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from atp_lean_gnn.premise_gnn import PremiseGNN


NUM_NODE_LABELS = 50
NUM_TACTICS = 100  # ← Required for PremiseGNN init
HIDDEN_DIM = 32


def _make_graph(num_nodes: int, num_edges: int, *, state_node_idx: int = 0) -> Data:
    """Build a minimal PyG Data object matching PremiseGNN's expected attributes."""
    data = Data()
    data.x = torch.randint(0, NUM_NODE_LABELS, (num_nodes,))
    data.node_type = torch.zeros(num_nodes, dtype=torch.long)
    data.is_bound = torch.zeros(num_nodes, dtype=torch.long)
    data.binder_depth = torch.zeros(num_nodes, dtype=torch.long)
    data.binder_kind = torch.zeros(num_nodes, dtype=torch.long)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    data.edge_index = torch.stack([src, dst], dim=0)
    data.state_node_index = torch.tensor([state_node_idx], dtype=torch.long)
    return data


def _make_lemma_graph(num_nodes: int, num_edges: int) -> Data:
    """Build a lemma graph (no state_node_index)."""
    data = _make_graph(num_nodes, num_edges)
    del data.state_node_index
    return data


class TestPremiseGNNContract:

    def test_encode_nodes_shape(self):
        """encode_nodes returns [num_nodes, hidden_dim]."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        graph = _make_graph(5, 4)
        batch = Batch.from_data_list([graph])
        node_embs = model.encode_nodes(batch)
        assert node_embs.shape == (5, HIDDEN_DIM)

    def test_readout_single_graph(self):
        """readout returns [1, hidden_dim] for a single graph."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        graph = _make_graph(5, 4, state_node_idx=0)
        batch = Batch.from_data_list([graph])
        node_embs = model.encode_nodes(batch)
        state_emb = model.readout(node_embs, batch)
        assert state_emb.shape == (1, HIDDEN_DIM)

    def test_readout_batch(self):
        """readout selects the correct per-graph State node using ptr offsets."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        # 3 graphs, 4 nodes each, state_node_idx=0 per graph.
        # After batching: global state nodes are at positions 0, 4, 8.
        graphs = [_make_graph(4, 3, state_node_idx=0) for _ in range(3)]
        batch = Batch.from_data_list(graphs)
        node_embs = model.encode_nodes(batch)  # [12, HIDDEN_DIM]
        state_embs = model.readout(node_embs, batch)  # [3, HIDDEN_DIM]
        assert state_embs.shape == (3, HIDDEN_DIM)
        # Verify each row matches the correct global node (0, 4, 8)
        assert torch.allclose(state_embs[0], node_embs[0])
        assert torch.allclose(state_embs[1], node_embs[4])
        assert torch.allclose(state_embs[2], node_embs[8])

    def test_readout_fallback_mean_pool(self):
        """readout falls back to mean pooling when state_node_index is absent (lemma graphs)."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        graph = _make_lemma_graph(5, 4)
        batch = Batch.from_data_list([graph])
        node_embs = model.encode_nodes(batch)
        state_emb = model.readout(node_embs, batch)
        assert state_emb.shape == (1, HIDDEN_DIM)

    def test_forward_convenience_wrapper(self):
        """forward() == readout(encode_nodes()) in shape."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        graph = _make_graph(5, 4)
        batch = Batch.from_data_list([graph])
        out = model(batch)
        assert out.shape == (1, HIDDEN_DIM)

    def test_sage_backbone(self):
        """backbone='sage' produces the same output shape as gatv2."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM,
            backbone="sage"
        )
        graph = _make_graph(5, 4)
        batch = Batch.from_data_list([graph])
        node_embs = model.encode_nodes(batch)
        state_emb = model.readout(node_embs, batch)
        assert node_embs.shape == (5, HIDDEN_DIM)
        assert state_emb.shape == (1, HIDDEN_DIM)

    def test_gradient_flow(self):
        """GNN parameters receive gradients through encode_nodes + readout."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        graph = _make_graph(4, 3)
        batch = Batch.from_data_list([graph])
        node_embs = model.encode_nodes(batch)
        state_emb = model.readout(node_embs, batch)
        state_emb.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_encode_lemma_cached(self):
        """encode_lemma_cached returns correct shape and populates the cache."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        graphs = [_make_lemma_graph(4, 3) for _ in range(3)]
        batch = Batch.from_data_list(graphs)
        lemma_ids = [10, 20, 30]
        cache: dict = {}

        result = model.encode_lemma_cached(batch, cache, lemma_ids)

        assert result.shape == (3, HIDDEN_DIM)
        assert set(cache.keys()) == {10, 20, 30}

    def test_encode_lemma_cached_reuses_cache(self):
        """encode_lemma_cached skips the GNN for already-cached lemmas.
        graph_batch must contain only the miss graphs (lemma 20 here).
        """
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        cached_vec = torch.randn(HIDDEN_DIM)
        cache = {10: cached_vec}

        # Only lemma 20 is a miss — pass a single-graph batch for it only
        graph = _make_lemma_graph(4, 3)
        batch = Batch.from_data_list([graph])

        result = model.encode_lemma_cached(batch, cache, [10, 20])

        assert result.shape == (2, HIDDEN_DIM)
        assert torch.allclose(result[0], cached_vec)
        assert 20 in cache

    def test_encode_state_cached_reuses_cache(self):
        """encode_state_cached skips the GNN for already-cached states."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        cached_vec = torch.randn(HIDDEN_DIM)
        cache = {10: cached_vec}

        # Only state 20 is a miss — pass a single-graph batch for it only
        batch = Batch.from_data_list([_make_graph(4, 3)])

        result = model.encode_state_cached(batch, cache, [10, 20])

        assert result.shape == (2, HIDDEN_DIM)
        assert torch.allclose(result[0], cached_vec)
        assert 20 in cache

    def test_encode_state_cached_rejects_miss_count_mismatch(self):
        """A batch that is not exactly the misses is an error, not silent misalignment.

        Passing every graph (rather than only the missing ones) used to index
        the wrong row of the readout and cache a wrong vector.
        """
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        cache = {10: torch.randn(HIDDEN_DIM)}
        # 2 graphs passed but only state 20 misses
        batch = Batch.from_data_list([_make_graph(4, 3), _make_graph(5, 4)])

        with pytest.raises(ValueError, match="cache misses"):
            model.encode_state_cached(batch, cache, [10, 20])

    def test_encode_lemma_cached_rejects_miss_count_mismatch(self):
        """encode_lemma_cached enforces the same batch/miss agreement."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        cache = {10: torch.randn(HIDDEN_DIM)}
        batch = Batch.from_data_list([_make_lemma_graph(4, 3), _make_lemma_graph(5, 4)])

        with pytest.raises(ValueError, match="cache misses"):
            model.encode_lemma_cached(batch, cache, [10, 20])

    def test_cached_vector_does_not_retain_batch_storage(self):
        """A cached vector must own its storage, not view into the whole batch.

        Rows of the readout are views; caching one unchanged would keep the
        entire [batch, hidden] tensor alive for as long as it stays cached.
        """
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        batch = Batch.from_data_list([_make_lemma_graph(4, 3) for _ in range(4)])
        cache: dict = {}

        model.encode_lemma_cached(batch, cache, [10, 20, 30, 40])

        for vec in cache.values():
            assert vec.untyped_storage().nbytes() == HIDDEN_DIM * vec.element_size()

    def test_tactic_embedding_exists(self):
        """PremiseGNN has tactic_embedding table."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        assert hasattr(model, "tactic_embedding")
        assert isinstance(model.tactic_embedding, torch.nn.Embedding)
        assert model.tactic_embedding.weight.shape == (NUM_TACTICS, HIDDEN_DIM)

    def test_query_proj_exists(self):
        """PremiseGNN has query_proj for contrastive learning."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        assert hasattr(model, "query_proj")
        assert isinstance(model.query_proj, torch.nn.Linear)
        assert model.query_proj.weight.shape == (HIDDEN_DIM, HIDDEN_DIM * 2)

    def test_key_proj_exists(self):
        """PremiseGNN has key_proj for contrastive learning."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        assert hasattr(model, "key_proj")
        assert isinstance(model.key_proj, torch.nn.Linear)
        assert model.key_proj.weight.shape == (HIDDEN_DIM, HIDDEN_DIM)

    def test_query_proj_ar_exists(self):
        """PremiseGNN has query_proj_ar for autoregressive selection."""
        model = PremiseGNN(
            num_node_labels=NUM_NODE_LABELS,
            num_tactics=NUM_TACTICS,
            hidden_dim=HIDDEN_DIM
        )
        assert hasattr(model, "query_proj_ar")
        assert isinstance(model.query_proj_ar, torch.nn.Linear)
        assert model.query_proj_ar.weight.shape == (HIDDEN_DIM, HIDDEN_DIM * 3)