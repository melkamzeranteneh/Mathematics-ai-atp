"""Tests for the Phase 2/3 harvest additions (HTPS/MCTS integration).

Covers: ``extract_critic_samples`` (status labels + soft W/N targets gated on
``visit_threshold``), the ``backup_values`` soft-bottom-out generalization, and
``extract_minimal_hypertree`` (step-minimal proof-tree mining under SOLVED nodes).
"""

from __future__ import annotations

import unittest

from maths_ai.data_models.proof_components import Goal, STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import NodeStatus, ProofHypergraph

from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import EdgeAction
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    backup_values,
    extract_critic_samples,
    extract_minimal_hypertree,
)


def _tac(name: str = "apply", p: float = 1.0) -> TacticCandidate:
    return TacticCandidate(tactic_name=name, arguments=[], probability=p)


def _stv() -> STV:
    return STV(strength=0.5, confidence=1.0)


class ExtractCriticSamplesTests(unittest.TestCase):
    def test_solved_and_dead_emit_hard_labels(self) -> None:
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        g.add_edge(g.root_id, _tac(), ranked_subgoals=[])  # QED ⇒ root SOLVED
        samples = extract_critic_samples(g, visit_threshold=4)
        by_goal = {s.goal: s.target for s in samples}
        self.assertEqual(by_goal["P"], 1.0)

        g2 = ProofHypergraph(Goal(expression="Q", hypotheses=[]))
        g2.mark_node_exhausted(g2.root_id)  # no edges + exhausted ⇒ DEAD
        self.assertEqual(g2.root.status, NodeStatus.DEAD)
        samples2 = extract_critic_samples(g2, visit_threshold=4)
        self.assertEqual({s.goal: s.target for s in samples2}["Q"], 0.0)

    def test_expanded_node_above_threshold_emits_w_over_n(self) -> None:
        g = ProofHypergraph(Goal(expression="P∧Q", hypotheses=[]))
        edge = g.add_edge(
            g.root_id, _tac(p=0.9),
            ranked_subgoals=[(Goal(expression="P", hypotheses=[]), _stv())],
        )
        edge.visit_stats.N = 4
        edge.visit_stats.W = 3.0
        samples = extract_critic_samples(g, visit_threshold=4)
        by_goal = {s.goal: s.target for s in samples}
        self.assertAlmostEqual(by_goal["P∧Q"], 0.75)

    def test_below_threshold_emits_nothing(self) -> None:
        g = ProofHypergraph(Goal(expression="P∧Q", hypotheses=[]))
        edge = g.add_edge(
            g.root_id, _tac(p=0.9),
            ranked_subgoals=[(Goal(expression="P", hypotheses=[]), _stv())],
        )
        edge.visit_stats.N = 3
        edge.visit_stats.W = 3.0
        samples = extract_critic_samples(g, visit_threshold=4)
        self.assertNotIn("P∧Q", {s.goal for s in samples})

    def test_soft_target_reads_the_max_prior_edge(self) -> None:
        # Two edges: the higher-prior one carries the statistics that become the
        # target; the other's are ignored even when larger.
        g = ProofHypergraph(Goal(expression="R", hypotheses=[]))
        low = g.add_edge(
            g.root_id, _tac("low", p=0.2),
            ranked_subgoals=[(Goal(expression="A", hypotheses=[]), _stv())],
        )
        high = g.add_edge(
            g.root_id, _tac("high", p=0.8),
            ranked_subgoals=[(Goal(expression="B", hypotheses=[]), _stv())],
        )
        low.visit_stats.N, low.visit_stats.W = 10, 10.0   # W/N = 1.0
        high.visit_stats.N, high.visit_stats.W = 10, 2.0  # W/N = 0.2 ← t*
        samples = extract_critic_samples(g, visit_threshold=4)
        self.assertAlmostEqual({s.goal: s.target for s in samples}["R"], 0.2)

    def test_unexpanded_open_node_is_skipped(self) -> None:
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        samples = extract_critic_samples(g, visit_threshold=0)
        self.assertEqual(samples, [])


class BackupValuesSoftTests(unittest.TestCase):
    def _expanded_graph(self, n: int, w: float):
        """root --e--> child; child unresolved with the given visit stats on e."""
        g = ProofHypergraph(Goal(expression="root", hypotheses=[]))
        edge = g.add_edge(
            g.root_id, _tac(p=0.9),
            ranked_subgoals=[(Goal(expression="child", hypotheses=[]), _stv())],
        )
        edge.visit_stats.N = n
        edge.visit_stats.W = w
        return g, edge

    def test_default_is_hard_zero(self) -> None:
        g, _edge = self._expanded_graph(n=10, w=8.0)
        vals = backup_values(g)
        # Unresolved child bottoms out at 0.0; root = max over edges of
        # product(children) = 0.0.
        self.assertEqual(vals[g.root_id], 0.0)

    def test_threshold_crossed_bottoms_out_at_w_over_n(self) -> None:
        g, edge = self._expanded_graph(n=10, w=8.0)
        # Give the child an expanded edge with its own consensus so both levels
        # carry soft evidence.
        child_id = edge.child_ids[0]
        child_edge = g.add_edge(
            child_id, _tac(p=0.5),
            ranked_subgoals=[(Goal(expression="leaf", hypotheses=[]), _stv())],
        )
        child_edge.visit_stats.N = 5
        child_edge.visit_stats.W = 3.0
        vals = backup_values(g, visit_threshold=4)
        # child: recursion floors at 0 (leaf has no evidence), lifted to its
        # own consensus 3/5. root: recursion gives product(0.6)=0.6, lifted to
        # its own max-prior edge's consensus 8/10.
        self.assertAlmostEqual(vals[child_id], 0.6)
        self.assertAlmostEqual(vals[g.root_id], 0.8)

    def test_below_threshold_reduces_to_hard_backup(self) -> None:
        g, edge = self._expanded_graph(n=2, w=2.0)
        vals_soft = backup_values(g, visit_threshold=4)
        vals_hard = backup_values(g)
        self.assertEqual(vals_soft, vals_hard)


class ExtractMinimalHypertreeTests(unittest.TestCase):
    def _solved_subgraph_under_failed_root(self):
        """root (unresolved) --e0--> mid; mid --e1(QED)--> [] (mid SOLVED)."""
        g = ProofHypergraph(Goal(expression="root", hypotheses=[]))
        e0 = g.add_edge(
            g.root_id, _tac("split", p=0.5),
            ranked_subgoals=[
                (Goal(expression="mid", hypotheses=[]), _stv()),
                (Goal(expression="stuck", hypotheses=[]), _stv()),
            ],
        )
        mid_id = e0.child_ids[0]
        e1 = g.add_edge(mid_id, _tac("trivial", p=0.9), ranked_subgoals=[])
        self.assertEqual(g.nodes[mid_id].status, NodeStatus.SOLVED)
        self.assertNotEqual(g.root.status, NodeStatus.SOLVED)
        return g, mid_id, e1

    def test_mine_all_finds_solved_interior_node(self) -> None:
        g, _mid_id, e1 = self._solved_subgraph_under_failed_root()
        actions = {e1.id: EdgeAction(tactic_id=3, arg_indices=(1, 2))}
        samples = extract_minimal_hypertree(g, actions, mine_all_solved_nodes=True)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].goal, "mid")
        self.assertEqual(samples[0].tactic_id, 3)
        self.assertEqual(samples[0].arg_indices, (1, 2))

    def test_root_only_mode_yields_nothing_for_failed_root(self) -> None:
        g, _mid_id, e1 = self._solved_subgraph_under_failed_root()
        actions = {e1.id: EdgeAction(tactic_id=3)}
        samples = extract_minimal_hypertree(g, actions, mine_all_solved_nodes=False)
        self.assertEqual(samples, [])

    def test_picks_the_step_minimal_edge(self) -> None:
        # Root SOLVED two ways: a 1-step QED and a 2-step route via a child.
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        long_edge = g.add_edge(
            g.root_id, _tac("longway", p=0.9),
            ranked_subgoals=[(Goal(expression="sub", hypotheses=[]), _stv())],
        )
        sub_qed = g.add_edge(long_edge.child_ids[0], _tac("trivial", p=0.9), ranked_subgoals=[])
        short_edge = g.add_edge(g.root_id, _tac("exact", p=0.1), ranked_subgoals=[])
        self.assertEqual(g.root.status, NodeStatus.SOLVED)
        actions = {
            long_edge.id: EdgeAction(tactic_id=1),
            sub_qed.id: EdgeAction(tactic_id=2),
            short_edge.id: EdgeAction(tactic_id=3),
        }
        samples = extract_minimal_hypertree(g, actions, mine_all_solved_nodes=False)
        # Only the 1-step edge is on the minimal tree.
        self.assertEqual([s.tactic_id for s in samples], [3])

    def test_edges_without_actions_are_skipped_but_traversed(self) -> None:
        # The minimal-tree edge at the root has no stored action (PLN-fallback
        # pseudo-edge); its child's edge does. Only the child's sample appears.
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        e0 = g.add_edge(
            g.root_id, _tac("PLN_fallback", p=0.5),
            ranked_subgoals=[(Goal(expression="sub", hypotheses=[]), _stv())],
        )
        e1 = g.add_edge(e0.child_ids[0], _tac("trivial", p=0.9), ranked_subgoals=[])
        self.assertEqual(g.root.status, NodeStatus.SOLVED)
        actions = {e1.id: EdgeAction(tactic_id=7)}
        samples = extract_minimal_hypertree(g, actions, mine_all_solved_nodes=False)
        self.assertEqual([s.tactic_id for s in samples], [7])

    def test_overlapping_trees_emit_each_edge_once(self) -> None:
        # mine_all: the mid node's minimal tree is a subset of the root's; the
        # shared QED edge must not produce a duplicate sample.
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        e0 = g.add_edge(
            g.root_id, _tac("step", p=0.5),
            ranked_subgoals=[(Goal(expression="mid", hypotheses=[]), _stv())],
        )
        e1 = g.add_edge(e0.child_ids[0], _tac("trivial", p=0.9), ranked_subgoals=[])
        actions = {
            e0.id: EdgeAction(tactic_id=1),
            e1.id: EdgeAction(tactic_id=2),
        }
        samples = extract_minimal_hypertree(g, actions, mine_all_solved_nodes=True)
        self.assertEqual(sorted(s.tactic_id for s in samples), [1, 2])


if __name__ == "__main__":
    unittest.main()
