"""Edge-selection scoring for multi-simulation proof search.

PUCT (predictor upper confidence bound applied to trees), as used by
HTPS: balances the empirical action value Q(g,t) against the policy
prior P(g,t), with an exploration bonus that decays as the edge
accumulates visits. Virtual loss enters through both terms — it lowers
Q (pending simulations count as losses) and inflates the visit count in
the bonus denominator — so simulations selected in the same batch spread
across different branches.

This module is also the single authority for the search-mode contract:
``resolve_search_params`` validates the (selection_policy, budget)
combination for both the reasoner constructor and the training config.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from maths_ai.hybrid_reasoner.hypergraph import EdgeVisitStats


def resolve_search_params(
    selection_policy: str,
    num_simulations: Optional[int],
    sim_batch_size: Optional[int],
    puct_c: Optional[float],
) -> Tuple[int, int, float]:
    """Validate the (policy, budget) combination and resolve None defaults.

    legacy: all three budget fields must be None — the legacy best-first
            loop has no simulation budget, so an explicit value would be
            silently ignored, and the run would proceed with settings the
            user believed were active. Returns ``(1, 1, 0.0)`` placeholders
            (never read by the legacy path).
    puct:   ``num_simulations`` is required (a PUCT search has no natural
            simulation count); ``sim_batch_size`` defaults to 4 and
            ``puct_c`` to 1.0.
    rp:     reserved for the deferred regularized-policy variant —
            ``NotImplementedError``.
    anything else: ``ValueError``.
    """
    if selection_policy == "legacy":
        explicit = {
            name: value
            for name, value in (
                ("num_simulations", num_simulations),
                ("sim_batch_size", sim_batch_size),
                ("puct_c", puct_c),
            )
            if value is not None
        }
        if explicit:
            raise ValueError(
                "selection_policy='legacy' runs the best-first loop, which has no "
                f"simulation budget; remove the explicit setting(s) {sorted(explicit)} "
                "or set selection_policy='puct'"
            )
        return 1, 1, 0.0
    if selection_policy == "puct":
        if num_simulations is None:
            raise ValueError(
                "selection_policy='puct' requires an explicit num_simulations "
                "(a PUCT search has no natural simulation count)"
            )
        return (
            num_simulations,
            4 if sim_batch_size is None else sim_batch_size,
            1.0 if puct_c is None else puct_c,
        )
    if selection_policy == "rp":
        raise NotImplementedError(
            "selection_policy='rp' (regularized policy) is reserved but not implemented"
        )
    raise ValueError(
        f"Unknown selection_policy {selection_policy!r} (use 'legacy' or 'puct')"
    )


def puct_score(stats: EdgeVisitStats, total_node_visits: int, c: float) -> float:
    """PUCT score for one tactic edge.

    The prior is ``stats.prior_prob`` — the tactic-head probability stamped
    onto the edge's ``EdgeVisitStats`` at creation time by
    ``ProofHypergraph.add_edge`` — read here rather than passed in, so no
    caller can substitute a live recomputation. ``total_node_visits`` is the
    sum of (N + virtual_loss) over all sibling edges at the parent node;
    ``c`` is the exploration constant.
    """
    return stats.Q + c * stats.prior_prob * math.sqrt(total_node_visits) / (
        1 + stats.N + stats.virtual_loss
    )
