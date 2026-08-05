"""Harvest actor-critic training targets from a finished ``ProofHypergraph`` (B3, AND-OR).

The hybrid reasoner's search produces an AND-OR proof graph whose solved/dead status is
back-propagated to the root. This module turns that graph into per-transition training
targets: a value target for the critic (from the AND-OR value backup) and a return for the
actor (shaped reward + bootstrapped successor value). The actor advantage
``Â = return − V_pred(s)`` is finished in the training loop, where the critic's prediction at
collection time is known.

Value backup (one-sided provability signal, per the approaches doc):
  * SOLVED node                → 1.0
  * DEAD node                  → 0.0
  * interior node (OR over tactics): ``max_edge AND-combine(children)``
  * AND-combine (all subgoals must close): product (default) or min of child values
  * unexpanded / unresolved leaf → 0.0 (not yet shown provable — keeps PLN out of the target)

HTPS-style soft targets (Phase 2): when the search ran multiple simulations, each edge
carries visit statistics (``N`` visits, ``W`` accumulated backup value). For an unresolved
node whose max-prior edge accumulated at least ``visit_threshold`` visits, ``W/N`` is a
search-consensus estimate of provability — ``extract_critic_samples`` emits it as a critic
regression target, and ``backup_values(..., visit_threshold=...)`` substitutes it for the
hard 0.0 the unresolved region would otherwise evaluate to. Both are pure visit statistics
and statuses; PLN never enters the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from maths_ai.data_models.proof_components import Goal, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import (
    EdgeStatus,
    NodeStatus,
    ProofHypergraph,
)

from .pln_reward import RewardConfig, edge_shaped_reward

if TYPE_CHECKING:
    from .pln_rl_training import EdgeAction


@dataclass(frozen=True)
class HarvestConfig:
    and_combine: str = "product"  # "product" or "min"
    unresolved_leaf_value: float = 0.0


@dataclass(frozen=True)
class CriticSample:
    """One node-level critic regression target (goal text → value in [0, 1]).

    Goal-keyed, not edge-keyed: the sample is re-featurized fresh at train time
    (deterministic under the fixed prepared vocab), so no featurizer state or
    autograd graph travels with it.
    """

    goal: str
    hypotheses: tuple[str, ...]
    target: float


@dataclass(frozen=True)
class TacticImitationSample:
    """One (goal, tactic) supervised pair mined from a proven subgraph (Phase 3).

    ``tactic_id`` and ``arg_indices`` come from the stored ``EdgeAction`` of the edge
    on the minimal proof hypertree; ``arg_indices`` keeps the sampled pointer
    positions verbatim (out-of-range/-1 entries retain the existing ignore
    semantics in ``forced_step``). Empty ``arg_indices`` is a tactic-only row.
    """

    goal: str
    hypotheses: tuple[str, ...]
    tactic_id: int
    arg_indices: tuple[int, ...] = ()


@dataclass
class HarvestedTransition:
    """One (state, action) training example extracted from the search graph."""
    node_id: int
    goal: Goal                 # state s (expression + hypotheses)
    tactic: TacticCandidate    # action a = (tactic, args)
    reward: float              # shaped per-edge reward r'  (r_term + γΦ(s')−Φ(s))
    children_value: float      # AND-combined backup value of the edge's subgoals (bootstrap)
    value_target: float        # AND-OR backup value of s  (critic regression target)
    return_: float             # reward + γ·children_value  (actor target return)
    edge_id: int = -1          # source hyperedge — the on-policy join key to EdgeAction


def _and_combine(values: list[float], cfg: HarvestConfig) -> float:
    if not values:
        return 1.0  # no subgoals ⇒ branch closed (QED)
    if cfg.and_combine == "min":
        return min(values)
    product = 1.0
    for v in values:
        product *= v
    return product


def backup_values(
    graph: ProofHypergraph,
    cfg: HarvestConfig | None = None,
    *,
    visit_threshold: int | None = None,
) -> dict[int, float]:
    """Compute the AND-OR value backup for every node (memoized, cycle-safe).

    ``visit_threshold`` (Phase 2): when set, an unresolved node whose max-prior edge
    accumulated at least that many visits bottoms out at the soft ``W/N`` consensus
    instead of ``cfg.unresolved_leaf_value``. Under ``selection_policy="legacy"`` no
    edge ever accumulates visits, so the default ``None`` (and any threshold, on a
    legacy-search graph) reduces to the original hard backup.
    """
    cfg = cfg or HarvestConfig()
    memo: dict[int, float] = {}
    in_progress: set[int] = set()

    def unresolved_value(node_id: int) -> float:
        if visit_threshold is not None:
            soft = _soft_target(graph, graph.nodes[node_id], visit_threshold)
            if soft is not None:
                return soft
        return cfg.unresolved_leaf_value

    def value(node_id: int) -> float:
        if node_id in memo:
            return memo[node_id]
        if node_id in in_progress:
            # Cycle (a subgoal identical to an ancestor): treat as unresolved.
            return unresolved_value(node_id)
        node = graph.nodes[node_id]
        if node.status == NodeStatus.SOLVED:
            memo[node_id] = 1.0
            return 1.0
        if node.status == NodeStatus.DEAD:
            memo[node_id] = 0.0
            return 0.0

        in_progress.add(node_id)
        edge_values: list[float] = []
        for edge_id in node.outgoing_edge_ids:
            edge = graph.edges[edge_id]
            if edge.status == EdgeStatus.DEAD:
                edge_values.append(0.0)
                continue
            child_values = [value(cid) for cid in edge.child_ids]
            edge_values.append(_and_combine(child_values, cfg))
        in_progress.discard(node_id)

        # OR over tactics; an unexpanded interior node (no edges) is unresolved.
        val = max(edge_values) if edge_values else unresolved_value(node_id)
        if visit_threshold is not None and edge_values:
            # Soft floor (Phase 2): the status-only recursion floors every
            # unresolved subtree at 0, erasing what the simulations learned.
            # The best-prior edge's W/N is the search's own value backup for
            # this node — take the max so accumulated consensus can lift the
            # hard floor but never overwrite stronger status evidence.
            soft = _soft_target(graph, node, visit_threshold)
            if soft is not None:
                val = max(val, soft)
        memo[node_id] = val
        return val

    return {node_id: value(node_id) for node_id in graph.nodes}


def _soft_target(graph: ProofHypergraph, node, visit_threshold: int) -> float | None:
    """Soft W/N critic target for an unresolved node, or ``None`` without evidence.

    ``t_star`` is the node's maximum-prior outgoing edge (the tactic the policy rated
    highest — HTPS regresses the critic toward the search's evaluation of that
    choice). The target is ``W/N`` of its visit statistics, defined only when
    ``N ≥ visit_threshold``; below the threshold the visit mean is noise, and an
    unexpanded node has no edges at all.
    """
    t_star = None
    for edge_id in node.outgoing_edge_ids:
        edge = graph.edges[edge_id]
        if t_star is None or edge.tactic.probability > t_star.tactic.probability:
            t_star = edge
    if t_star is None or t_star.visit_stats.N < visit_threshold:
        return None
    return t_star.visit_stats.W / t_star.visit_stats.N


def extract_critic_samples(
    graph: ProofHypergraph,
    *,
    visit_threshold: int,
) -> list[CriticSample]:
    """Node-level critic targets from statuses and visit statistics (Phase 2).

    * SOLVED node → 1.0; DEAD node → 0.0 (the resolved labels the on-policy path
      already learns from at terminals — repeated here so the decoupled step sees
      both ends of the scale, not only soft mid-range values).
    * Any other node (EXPANDED, or OPEN without edges) → the soft ``W/N`` target of
      its max-prior edge when that edge has ``N ≥ visit_threshold`` visits; skipped
      otherwise (insufficient evidence is no label, not a 0 label).

    PLN never enters the target.
    """
    samples: list[CriticSample] = []
    for node in graph.nodes.values():
        if node.status == NodeStatus.SOLVED:
            target = 1.0
        elif node.status == NodeStatus.DEAD:
            target = 0.0
        else:
            soft = _soft_target(graph, node, visit_threshold)
            if soft is None:
                continue
            target = soft
        samples.append(
            CriticSample(
                goal=node.goal.expression,
                hypotheses=tuple(node.goal.hypotheses),
                target=target,
            )
        )
    return samples


def extract_minimal_hypertree(
    graph: ProofHypergraph,
    edge_actions: Mapping[int, "EdgeAction"],
    *,
    mine_all_solved_nodes: bool = True,
) -> list[TacticImitationSample]:
    """Mine (goal, tactic) imitation pairs from the minimal proof hypertrees (Phase 3).

    For each SOLVED node (every one when ``mine_all_solved_nodes`` — including inside
    searches whose root failed — else the root only, and only if it is SOLVED):
    recursively pick, among the node's SOLVED edges, the one minimizing the total
    downstream step count (number of edges in the proof tree beneath it, memoized over
    the shared subgraph), and emit one sample per edge on that minimal tree.

    Only edges present in ``edge_actions`` (keyed ``edge.id → EdgeAction``) yield
    samples: an edge without a stored policy action is a PLN-fallback pseudo-edge or
    outside the on-policy filter, and neither is a valid imitation target. The
    traversal still descends through such edges — their children's own edges may
    carry actions. Overlapping minimal trees (a solved node inside another solved
    node's tree) emit each edge once.

    Minimality is step count only: ``TacticOutcome`` records no timing, so a
    ``tactic_cpu_time`` criterion would need executor instrumentation that does not
    exist (future work).
    """
    steps_memo: dict[int, float] = {}
    in_progress: set[int] = set()

    def min_steps(node_id: int) -> float:
        """Minimal proof-tree edge count under a SOLVED node (inf if none closes)."""
        if node_id in steps_memo:
            return steps_memo[node_id]
        if node_id in in_progress:
            return float("inf")
        node = graph.nodes[node_id]
        if node.status != NodeStatus.SOLVED:
            return float("inf")
        in_progress.add(node_id)
        best = float("inf")
        for edge_id in node.outgoing_edge_ids:
            edge = graph.edges[edge_id]
            if edge.status != EdgeStatus.SOLVED:
                continue
            best = min(best, 1.0 + sum(min_steps(cid) for cid in edge.child_ids))
        in_progress.discard(node_id)
        steps_memo[node_id] = best
        return best

    def best_edge(node_id: int) -> int | None:
        """The SOLVED edge achieving ``min_steps`` at this node."""
        node = graph.nodes[node_id]
        chosen_id = None
        chosen_steps = float("inf")
        for edge_id in node.outgoing_edge_ids:
            edge = graph.edges[edge_id]
            if edge.status != EdgeStatus.SOLVED:
                continue
            steps = 1.0 + sum(min_steps(cid) for cid in edge.child_ids)
            if steps < chosen_steps:
                chosen_id = edge_id
                chosen_steps = steps
        return chosen_id

    if mine_all_solved_nodes:
        roots = [n.id for n in graph.nodes.values() if n.status == NodeStatus.SOLVED]
    else:
        roots = [graph.root_id] if graph.root.status == NodeStatus.SOLVED else []

    samples: list[TacticImitationSample] = []
    emitted: set[int] = set()

    def walk(node_id: int) -> None:
        edge_id = best_edge(node_id)
        if edge_id is None or edge_id in emitted:
            return
        emitted.add(edge_id)
        edge = graph.edges[edge_id]
        action = edge_actions.get(edge_id)
        if action is not None:
            node = graph.nodes[node_id]
            samples.append(
                TacticImitationSample(
                    goal=node.goal.expression,
                    hypotheses=tuple(node.goal.hypotheses),
                    tactic_id=action.tactic_id,
                    arg_indices=tuple(action.arg_indices),
                )
            )
        for cid in edge.child_ids:
            walk(cid)

    for root_id in roots:
        walk(root_id)
    return samples


def extract_transitions(
    graph: ProofHypergraph,
    reward_cfg: RewardConfig | None = None,
    harvest_cfg: HarvestConfig | None = None,
    *,
    edge_ids: list[int] | None = None,
) -> list[HarvestedTransition]:
    """Turn the search graph into per-transition training targets.

    One ``HarvestedTransition`` per hyperedge (an applied tactic). ``edge_ids`` restricts to a
    subset — the on-policy training loop passes only the edges whose tactic was sampled from
    the current policy, so the collected targets are on-policy. Without it, every edge is
    harvested (useful for tests and off-policy analysis).
    """
    reward_cfg = reward_cfg or RewardConfig()
    harvest_cfg = harvest_cfg or HarvestConfig()
    values = backup_values(graph, harvest_cfg)

    chosen = edge_ids if edge_ids is not None else list(graph.edges.keys())
    transitions: list[HarvestedTransition] = []
    for edge_id in chosen:
        edge = graph.edges[edge_id]
        parent = graph.nodes[edge.source_id]
        reward = edge_shaped_reward(edge, graph, reward_cfg)
        child_values = [values[cid] for cid in edge.child_ids]
        children_value = _and_combine(child_values, harvest_cfg)
        return_ = reward + reward_cfg.gamma * children_value
        transitions.append(
            HarvestedTransition(
                node_id=parent.id,
                goal=parent.goal,
                tactic=edge.tactic,
                reward=reward,
                children_value=children_value,
                value_target=values[parent.id],
                return_=return_,
                edge_id=edge_id,
            )
        )
    return transitions
