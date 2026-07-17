"""Live smoke harness for the on-policy RL loop (manual — needs Lean + petta).

Run:  uv run python -m maths_ai.gnn_inference.scripts.rl_smoke

What it validates end-to-end with a REAL Pantograph server and a REAL petta binary:
  collect: RLHybridReasoner samples from a fresh (untrained) actor-critic, decodes each
  draw into a Lean tactic, the PantographExecutor applies it, PLN scores the survivors,
  and the search returns an RLSearchResult.
  train:   train_step_onpolicy harvests the on-policy edges + failure records and takes
  exactly one gradient step.

The policy is untrained, so most draws are rejected by Lean — that IS the expected
signal path (failure records dominate). To make the QED branch reachable within a few
draws, the actor's base head is biased toward "intro"/"exact"/"assumption" on the seed
goal ``∀ (p : Prop), p → p``. This validates control flow and reward plumbing only;
proving quality waits on trained checkpoints.
"""

from __future__ import annotations

import asyncio

import torch
from torch.optim import AdamW

from pantograph.server import Server

from maths_ai.data_models.proof_components import Goal
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor

from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import ActorCriticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    goal_to_state,
    train_step_onpolicy,
)
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner


TACTIC_VOCAB = {
    "intro": 0,
    "exact": 1,
    "assumption": 2,
    "apply": 3,
    "constructor": 4,
    "trivial": 5,
}

# Seed theorem: identity. `intro h` then `exact h` (or `assumption`) closes it.
SEED_GOAL = "p → p"
SEED_HYPS = ["p : Prop"]


def build_model(node_vocab: dict[str, int]) -> ActorCriticWithArgsClassifier:
    model = ActorCriticWithArgsClassifier(
        num_node_labels=len(node_vocab),
        num_tactics=len(TACTIC_VOCAB),
        hidden_dim=32,
        num_layers=2,
        dropout=0.1,
        max_args=1,
    )
    # Bias the fresh policy toward the tactics that can actually close the seed goal,
    # so the QED branch fires within a handful of i.i.d. draws. The base head's random
    # logits are O(1), so +6/+4 makes intro/assumption dominate the softmax.
    with torch.no_grad():
        model.actor.base.bias[TACTIC_VOCAB["intro"]] += 6.0
        model.actor.base.bias[TACTIC_VOCAB["assumption"]] += 4.0
    return model


async def run_smoke() -> None:
    torch.manual_seed(0)

    seed = Goal(expression=SEED_GOAL, hypotheses=SEED_HYPS)
    node_vocab = build_vocab([proof_state_to_dag(goal_to_state(seed))])
    model = build_model(node_vocab)

    # Server.create() binds the subprocess pipes to THIS event loop; the sync Server()
    # constructor would bind them to its own internal loop and every later await from
    # asyncio.run's loop would fail with "attached to a different loop".
    print("[rl_smoke] starting Pantograph server…")
    server = await Server.create()
    executor = PantographExecutor(server)

    reasoner = RLHybridReasoner(
        model,
        node_vocab,
        TACTIC_VOCAB,
        executor=executor,
        top_k_tactics=4,
        max_depth=4,
        max_nodes=30,
    )

    print(f"[rl_smoke] collect: proving `{SEED_GOAL}` with hyps {SEED_HYPS}")
    result = await reasoner.prove(SEED_GOAL, hypotheses=SEED_HYPS)

    graph = result.graph
    print(
        f"[rl_smoke] search done: solved={graph.is_solved()} nodes={len(graph.nodes)} "
        f"edges={len(graph.edges)} on-policy edges={len(result.edge_actions)} "
        f"failures={len(result.failure_actions)}"
    )
    assert result.edge_actions or result.failure_actions, (
        "no on-policy signal collected — neither an accepted edge nor a failure record"
    )

    optimizer = AdamW(model.parameters(), lr=1e-3)
    metrics = train_step_onpolicy(
        model, optimizer, [result], reasoner.dag_featurize_data,
        reward_cfg=RewardConfig(step_penalty=0.01),
        bc_weight=0.0,
    )
    print(f"[rl_smoke] train step metrics: {metrics}")
    assert metrics.get("num_transitions", 0.0) + metrics.get("num_failures", 0.0) > 0
    assert all(v == v for v in metrics.values()), "NaN in training metrics"
    print("[rl_smoke] OK — collect → harvest → one on-policy gradient step completed.")


def main() -> None:
    asyncio.run(run_smoke())


if __name__ == "__main__":
    main()
