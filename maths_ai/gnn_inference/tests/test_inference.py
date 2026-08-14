from __future__ import annotations

import numpy as np
import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.inference import (
    InferencePipeline,
    _top_tactic_candidates,
)
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.premise_gnn import PremiseGNN
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer


HIDDEN_DIM = 32
STATE_A = "h : p ∨ q\n⊢ q ∨ p"
STATE_B = "⊢ p → p"

# The cache sits above the encoder choice, so it must behave the same whether
# embeddings come from PremiseGNN or from the classifier's own backbone.
BOTH_ENCODERS = pytest.mark.parametrize(
    "with_premise_gnn", [False, True], ids=["backbone", "premise_gnn"]
)


def _make_pipeline(
    *,
    state_cache_size: int = 128,
    with_premise_gnn: bool = False,
) -> InferencePipeline:
    """A small CPU pipeline over a 3-lemma index, deterministic in eval mode."""
    import faiss

    torch.manual_seed(0)
    tactic_vocab = {"exact": 0, "apply": 1, "intro": 2}

    model = TacticWithArgsClassifier(
        num_node_labels=8,
        num_tactics=len(tactic_vocab),
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        dropout=0.0,
        use_node_type=True,
        max_args=3,
    )
    scorer = PremiseScorer(hidden_dim=HIDDEN_DIM, mode="dot")

    premise_gnn = None
    if with_premise_gnn:
        premise_gnn = PremiseGNN(
            num_node_labels=8,
            num_tactics=len(tactic_vocab),
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            heads=4,
            dropout=0.0,
        )

    lemma_vectors = np.random.RandomState(0).randn(3, HIDDEN_DIM).astype(np.float32)
    index = faiss.IndexFlatIP(HIDDEN_DIM)
    index.add(lemma_vectors)

    return InferencePipeline(
        model=model,
        scorer=scorer,
        lemma_index=LemmaIndex(index, [1, 2, 3], lemma_vectors),
        node_vocab={"<UNK>": 0},
        tactic_vocab=tactic_vocab,
        device=torch.device("cpu"),
        k=3,
        premise_gnn=premise_gnn,
        state_cache_size=state_cache_size,
    )


def test_top_tactic_candidates_returns_sorted_top_k() -> None:
    tactic_probs = torch.tensor([0.1, 0.5, 0.4], dtype=torch.float32)
    id_to_tactic = {0: "simp", 1: "rw", 2: "exact"}

    candidates = _top_tactic_candidates(tactic_probs, id_to_tactic, top_k=2)

    assert [(item["tactic_id"], item["tactic_name"], item["probability"]) for item in candidates] == [
        (1, "rw", 0.5),
        (2, "exact", 0.4),
    ]


def test_top_tactic_candidates_caps_at_vocab_size() -> None:
    tactic_probs = torch.tensor([0.2, 0.3], dtype=torch.float32)
    id_to_tactic = {0: "simp", 1: "rw"}

    candidates = _top_tactic_candidates(tactic_probs, id_to_tactic, top_k=10)

    assert len(candidates) == 2


@BOTH_ENCODERS
def test_repeated_state_hits_the_cache(with_premise_gnn: bool) -> None:
    """Re-expanding the same goal reuses the encode instead of redoing it."""
    pipeline = _make_pipeline(with_premise_gnn=with_premise_gnn)

    first = pipeline.predict_tactic_result(STATE_A, top_k=3)
    assert pipeline.state_cache_stats == {
        "hits": 0,
        "misses": 1,
        "size": 1,
        "capacity": 128,
    }

    second = pipeline.predict_tactic_result(STATE_A, top_k=3)
    stats = pipeline.state_cache_stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1

    # The cache must not change what is predicted
    assert first.predicted_tactic == second.predicted_tactic
    assert first.tactic_probabilities == second.tactic_probabilities
    assert first.selected_arguments == second.selected_arguments


@BOTH_ENCODERS
def test_cache_hit_returns_the_same_bundle_without_recomputing(with_premise_gnn: bool) -> None:
    """The point of the cache: a hit hands back the identical encode.

    Object identity is the check that matters — if any of the parse, the DAG
    build, the GNN forward pass or the FAISS lookup had re-run, these would be
    equal-valued but distinct objects.
    """
    pipeline = _make_pipeline(with_premise_gnn=with_premise_gnn)

    first = pipeline._encode_state(STATE_A)
    second = pipeline._encode_state(STATE_A)

    assert second is first, "cache hit rebuilt the encoded state"
    assert second.dag is first.dag
    assert second.state_emb is first.state_emb
    assert second.pool is first.pool
    assert pipeline.state_cache_stats["hits"] == 1

    # A different goal is genuinely encoded afresh
    other = pipeline._encode_state(STATE_B)
    assert other is not first
    assert pipeline.state_cache_stats["misses"] == 2


@BOTH_ENCODERS
def test_cached_and_uncached_pipelines_agree(with_premise_gnn: bool) -> None:
    """Caching is an optimisation only — disabling it changes nothing."""
    cached = _make_pipeline(
        state_cache_size=8, with_premise_gnn=with_premise_gnn
    ).predict_tactic_result(STATE_A, top_k=3)
    uncached = _make_pipeline(
        state_cache_size=0, with_premise_gnn=with_premise_gnn
    ).predict_tactic_result(STATE_A, top_k=3)

    assert cached.predicted_tactic == uncached.predicted_tactic
    assert cached.tactic_probabilities == uncached.tactic_probabilities
    assert cached.selected_arguments == uncached.selected_arguments


def test_state_cache_size_zero_disables_caching() -> None:
    pipeline = _make_pipeline(state_cache_size=0)

    pipeline.predict_tactic_result(STATE_A)
    pipeline.predict_tactic_result(STATE_A)

    assert pipeline.state_cache_stats == {
        "hits": 0,
        "misses": 0,
        "size": 0,
        "capacity": 0,
    }


def test_state_cache_evicts_least_recently_used() -> None:
    pipeline = _make_pipeline(state_cache_size=1)

    pipeline.predict_tactic_result(STATE_A)
    pipeline.predict_tactic_result(STATE_B)

    assert pipeline.state_cache_stats["size"] == 1
    # STATE_A was evicted, so seeing it again is a miss
    pipeline.predict_tactic_result(STATE_A)
    assert pipeline.state_cache_stats["hits"] == 0
    assert pipeline.state_cache_stats["misses"] == 3


def test_clear_state_cache_resets_counters_and_entries() -> None:
    pipeline = _make_pipeline()

    pipeline.predict_tactic_result(STATE_A)
    pipeline.clear_state_cache()

    assert pipeline.state_cache_stats == {
        "hits": 0,
        "misses": 0,
        "size": 0,
        "capacity": 128,
    }

    pipeline.predict_tactic_result(STATE_A)
    assert pipeline.state_cache_stats["misses"] == 1
