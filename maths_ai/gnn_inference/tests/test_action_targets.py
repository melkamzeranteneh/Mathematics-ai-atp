from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.action_targets import (
    ActionTargetAuditConfig,
    ActionTargetError,
    compile_action_trace,
    run_action_target_audit,
)
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.preparation import ActionTraceCache, SExprCache


def _trace(*, terms=None, syntax_args=None, local_indices=(3, 4)) -> dict:
    normalized_terms = []
    for index, term in enumerate([] if terms is None else terms):
        normalized_terms.append({"source_start": index * 10, **term})
    normalized_syntax_args = []
    for index, argument in enumerate([] if syntax_args is None else syntax_args):
        normalized_syntax_args.append({"source_start": 100 + index * 10, **argument})
    return {
        "terms": normalized_terms,
        "syntax_args": normalized_syntax_args,
        "local_context": [
            {"context_index": index, "user_name": f"v{index}"}
            for index in local_indices
        ],
    }


class ActionTargetCompilerTests(unittest.TestCase):
    def test_compiles_application_of_two_local_references(self) -> None:
        target = compile_action_trace(
            _trace(terms=[{"action_sexp": "(:app (:local FV3) (:local FV4))"}])
        )

        self.assertEqual(
            target.operations,
            (
                {"op": "TERM_START"},
                {"op": "APP", "arity": 1},
                {"op": "LOCAL", "context_index": 3},
                {"op": "LOCAL", "context_index": 4},
                {"op": "TERM_END"},
                {"op": "STOP"},
            ),
        )

    def test_compiles_constructor_nested_application_and_fresh_name(self) -> None:
        target = compile_action_trace(
            _trace(
                terms=[
                    {
                        "action_sexp": (
                            "(:app (:ctor And.intro) (:local FV3) "
                            "(:app (:global Demo.f) (:local FV4)))"
                        )
                    }
                ],
                syntax_args=[{"role": "fresh_name", "source": "x"}],
            )
        )

        self.assertEqual(
            [operation["op"] for operation in target.operations],
            [
                "TERM_START",
                "APP",
                "CONSTRUCTOR",
                "LOCAL",
                "APP",
                "GLOBAL",
                "LOCAL",
                "TERM_END",
                "FRESH_NAME",
                "STOP",
            ],
        )
        self.assertTrue(target.has_payload)

    def test_compiles_literal_with_spaces_and_projection(self) -> None:
        target = compile_action_trace(
            _trace(
                terms=[
                    {"action_sexp": '(:lit "hello world")'},
                    {"action_sexp": "(:proj Demo.Pair 1 (:local FV3))"},
                ]
            )
        )

        self.assertIn({"op": "LITERAL", "value": '"hello world"'}, target.operations)
        self.assertIn(
            {"op": "PROJECTION", "type_name": "Demo.Pair", "index": 1},
            target.operations,
        )

    def test_merges_term_and_syntax_arguments_in_source_order(self) -> None:
        target = compile_action_trace(
            _trace(
                terms=[{"source_start": 20, "action_sexp": "(:local FV3)"}],
                syntax_args=[
                    {"source_start": 10, "role": "fresh_name", "source": "x"}
                ],
            )
        )

        self.assertEqual(
            [operation["op"] for operation in target.operations],
            ["FRESH_NAME", "TERM_START", "LOCAL", "TERM_END", "STOP"],
        )

    def test_compiles_scoped_expression_forms(self) -> None:
        target = compile_action_trace(
            _trace(
                terms=[
                    {
                        "action_sexp": (
                            "(:lambda x (:sort Type) "
                            "(:let y (:sort Type) (:lit 1) (:bound 0)))"
                        )
                    },
                    {
                        "action_sexp": (
                            "(:forall x (:sort Prop) "
                            "(:proj Demo.Pair 0 (:metavar)))"
                        )
                    },
                ]
            )
        )

        self.assertEqual(
            [operation["op"] for operation in target.operations],
            [
                "TERM_START",
                "LAMBDA",
                "SORT",
                "LET",
                "SORT",
                "LITERAL",
                "BOUND",
                "TERM_END",
                "TERM_START",
                "FORALL",
                "SORT",
                "PROJECTION",
                "METAVAR",
                "TERM_END",
                "STOP",
            ],
        )

    def test_rejects_local_reference_absent_from_context(self) -> None:
        with self.assertRaisesRegex(ActionTargetError, "absent from local_context") as caught:
            compile_action_trace(
                _trace(terms=[{"action_sexp": "(:local FV99)"}])
            )
        self.assertEqual(caught.exception.code, "unknown_local_reference")

    def test_empty_trace_compiles_to_stop_but_is_not_payload(self) -> None:
        target = compile_action_trace(_trace())
        self.assertEqual(target.operations, ({"op": "STOP"},))
        self.assertFalse(target.has_payload)


class ActionTargetAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("maths_ai/gnn_inference/tests/_tmp_action_targets")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.cache_root = self.root / "cache"
        self.output_dir = self.root / "report"

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_audit_validates_digest_and_writes_targets(self) -> None:
        row = DatasetRow(
            state="h : P\n⊢ P",
            target_state="no goals to be solved",
            theorem="Demo.result",
            tactic="exact h",
            split="train",
            row_index=7,
            dataset_name="fake/dataset",
            repo_url="https://example.invalid/mathlib4",
            repo_commit="commit",
            file_path="Mathlib/Demo.lean",
        )
        raw_cache = SExprCache(self.cache_root, project_path="")
        raw_record = {
            "schema_version": SExprCache.SCHEMA_VERSION,
            "extractor_version": SExprCache.EXTRACTOR_VERSION,
            "dataset": row.dataset_name,
            "split": row.split,
            "row_index": row.row_index,
            "theorem": row.theorem,
            "state_sha256": SExprCache.row_state_sha256(row),
            "tactic_sha256": SExprCache.row_tactic_sha256(row),
            "target_state_sha256": SExprCache.row_target_state_sha256(row),
            "repo_commit": row.repo_commit,
            "file_path": row.file_path,
            "goal_sexp": "(:c P)",
            "hyp_sexps": [],
        }
        raw_cache.save("train", 7, raw_record)
        action_cache = ActionTraceCache(self.cache_root)
        action_cache.save(
            "train",
            7,
            {
                "schema_version": ActionTraceCache.SCHEMA_VERSION,
                "extractor_version": ActionTraceCache.EXTRACTOR_VERSION,
                "raw_record_sha256": ActionTraceCache.raw_record_sha256(raw_record),
                "theorem": row.theorem,
                "tactic": row.tactic,
                "terms": [{"source_start": 10, "action_sexp": "(:local FV3)"}],
                "syntax_args": [],
                "local_context": [{"context_index": 3, "user_name": "h"}],
            },
        )

        summary = run_action_target_audit(
            ActionTargetAuditConfig(
                cache_root=self.cache_root,
                output_dir=self.output_dir,
            )
        )

        train = summary["splits"]["train"]
        self.assertEqual(train["compiled_count"], 1)
        self.assertEqual(train["payload_row_count"], 1)
        self.assertIn('"op": "LOCAL"', (self.output_dir / "targets.jsonl").read_text())
        self.assertTrue((self.output_dir / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
