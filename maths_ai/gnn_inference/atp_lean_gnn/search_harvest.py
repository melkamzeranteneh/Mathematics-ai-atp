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
"""

from __future__ import annotations

from dataclasses import dataclass

from maths_ai.data_models.proof_components import Goal, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import (
    EdgeStatus,
    NodeStatus,
    ProofHypergraph,
)

from .pln_reward import RewardConfig, edge_shaped_reward


@dataclass(frozen=True)
class HarvestConfig:
    and_combine: str = "product"  # "product" or "min"
    unresolved_leaf_value: float = 0.0


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
) -> dict[int, float]:
    """Compute the AND-OR value backup for every node (memoized, cycle-safe)."""
    cfg = cfg or HarvestConfig()
    memo: dict[int, float] = {}
    in_progress: set[int] = set()

    def value(node_id: int) -> float:
        if node_id in memo:
            return memo[node_id]
        if node_id in in_progress:
            # Cycle (a subgoal identical to an ancestor): treat as unresolved.
            return cfg.unresolved_leaf_value
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
        val = max(edge_values) if edge_values else cfg.unresolved_leaf_value
        memo[node_id] = val
        return val

    return {node_id: value(node_id) for node_id in graph.nodes}


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
