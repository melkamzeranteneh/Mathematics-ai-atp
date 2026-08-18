from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.action_targets import (
    ActionTargetAuditConfig,
    ActionTargetError,
    compile_action_trace,
    compile_source_syntax_trace,
    run_action_target_audit,
)
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.preparation import (
    ActionTraceCache,
    SExprCache,
    SourceSyntaxTraceCache,
)


def _node(kind: str, *children: dict, source: str = "") -> dict:
    return {"tag": "node", "kind": kind, "source": source, "children": list(children)}


def _atom(value: str, start: int = 0) -> dict:
    return {
        "tag": "atom",
        "source": value,
        "sourceStart": start,
        "sourceEnd": start + len(value),
    }


def _identifier(source: str, start: int = 0, **annotations) -> dict:
    return {
        "tag": "identifier",
        "source": source,
        "sourceStart": start,
        "sourceEnd": start + len(source),
        **annotations,
    }


def _source_trace(root: dict, *, syntax_args=None, local_indices=(0, 1)) -> dict:
    return {
        "source_syntax": root,
        "syntax_args": [] if syntax_args is None else list(syntax_args),
        "local_context": [
            {"context_index": index, "user_name": f"v{index}"}
            for index in local_indices
        ],
    }


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


class SourceSyntaxTargetCompilerTests(unittest.TestCase):
    def test_compiles_annotated_tactic_syntax_in_source_order(self) -> None:
        target = compile_source_syntax_trace(
            _source_trace(
                _node(
                    "Lean.Parser.Tactic.exact",
                    _atom("exact", 0),
                    _node(
                        "Lean.Parser.Term.anonymousCtor",
                        _identifier(
                            "h", 7, semanticRole="local", contextIndex=1
                        ),
                        _identifier(
                            "Demo.f",
                            10,
                            semanticRole="global",
                            name="Demo.f",
                        ),
                    ),
                )
            )
        )

        self.assertEqual(
            target.operations,
            (
                {"op": "NODE", "kind": "Lean.Parser.Tactic.exact", "arity": 2},
                {"op": "ATOM", "value": "exact"},
                {"op": "NODE", "kind": "Lean.Parser.Term.anonymousCtor", "arity": 2},
                {"op": "LOCAL", "context_index": 1},
                {"op": "GLOBAL", "name": "Demo.f"},
                {"op": "STOP"},
            ),
        )
        self.assertEqual(target.local_reference_count, 1)
        self.assertTrue(target.has_payload)
        self.assertTrue(target.is_reference_resolved)

    def test_nested_tactic_block_stays_compact(self) -> None:
        target = compile_source_syntax_trace(
            _source_trace(
                _node(
                    "Lean.Parser.Tactic.exact",
                    _atom("exact", 0),
                    _node(
                        "Lean.Parser.Term.byTactic",
                        _atom("by", 6),
                        _node(
                            "Lean.Parser.Tactic.linearCombination",
                            _atom("linear_combination", 9),
                            _identifier(
                                "h", 28, semanticRole="local", contextIndex=0
                            ),
                        ),
                    ),
                )
            )
        )

        # The elaborated kernel proof of the nested block never appears, so the
        # sequence stays the size of the written source.
        self.assertEqual(
            [operation["op"] for operation in target.operations],
            ["NODE", "ATOM", "NODE", "ATOM", "NODE", "ATOM", "LOCAL", "STOP"],
        )
        self.assertEqual(target.node_count, 3)
        self.assertEqual(target.atom_count, 3)

    def test_constructor_and_scoped_local_are_distinct_operations(self) -> None:
        target = compile_source_syntax_trace(
            _source_trace(
                _node(
                    "Lean.Parser.Tactic.refine",
                    _identifier(
                        "And.intro",
                        0,
                        semanticRole="constructor",
                        name="And.intro",
                    ),
                    _identifier("x", 10, semanticRole="scoped_local"),
                )
            )
        )

        self.assertIn({"op": "CONSTRUCTOR", "name": "And.intro"}, target.operations)
        self.assertIn({"op": "SCOPED_LOCAL", "source": "x"}, target.operations)
        self.assertEqual(target.scoped_local_count, 1)
        self.assertTrue(target.is_reference_resolved)

    def test_fresh_binder_name_is_recognised_by_byte_range(self) -> None:
        target = compile_source_syntax_trace(
            _source_trace(
                _node(
                    "Lean.Parser.Tactic.intro",
                    _atom("intro", 0),
                    _identifier("hx", 6),
                ),
                syntax_args=[
                    {
                        "role": "fresh_name",
                        "source": "hx",
                        "source_start": 6,
                        "source_end": 8,
                    }
                ],
            )
        )

        self.assertIn({"op": "FRESH_NAME", "source": "hx"}, target.operations)
        self.assertEqual(target.fresh_name_count, 1)
        self.assertEqual(target.unannotated_identifier_count, 0)
        self.assertTrue(target.is_reference_resolved)

    def test_unannotated_identifier_keeps_spelling_without_claiming_a_reference(
        self,
    ) -> None:
        target = compile_source_syntax_trace(
            _source_trace(_node("Lean.Parser.Tactic.simp", _identifier("foo", 5)))
        )

        self.assertIn({"op": "IDENTIFIER", "source": "foo"}, target.operations)
        self.assertEqual(target.unannotated_identifier_count, 1)
        self.assertFalse(target.is_reference_resolved)

    def test_missing_syntax_is_counted_instead_of_silently_dropped(self) -> None:
        target = compile_source_syntax_trace(
            _source_trace(_node("Lean.Parser.Tactic.rwSeq", {"tag": "missing"}))
        )

        self.assertIn({"op": "MISSING"}, target.operations)
        self.assertEqual(target.missing_count, 1)
        self.assertFalse(target.is_reference_resolved)

    def test_empty_optional_positions_are_counted_as_structural_overhead(self) -> None:
        target = compile_source_syntax_trace(
            _source_trace(_node("Lean.Parser.Tactic.rwSeq", _node("null")))
        )

        self.assertEqual(target.node_count, 2)
        self.assertEqual(target.empty_null_node_count, 1)
        self.assertFalse(target.has_payload)

    def test_rejects_local_reference_absent_from_local_context(self) -> None:
        with self.assertRaises(ActionTargetError) as caught:
            compile_source_syntax_trace(
                _source_trace(
                    _identifier("h", 0, semanticRole="local", contextIndex=99)
                )
            )
        self.assertEqual(caught.exception.code, "unknown_local_reference")

    def test_rejects_unknown_semantic_role_and_unknown_tag(self) -> None:
        with self.assertRaises(ActionTargetError) as role_error:
            compile_source_syntax_trace(
                _source_trace(_identifier("h", 0, semanticRole="mystery"))
            )
        self.assertEqual(role_error.exception.code, "unsupported_semantic_role")

        with self.assertRaises(ActionTargetError) as tag_error:
            compile_source_syntax_trace(_source_trace({"tag": "mystery"}))
        self.assertEqual(tag_error.exception.code, "unsupported_source_syntax_tag")

    def test_deeply_nested_syntax_does_not_exhaust_the_python_stack(self) -> None:
        root: dict = _identifier("h", 0, semanticRole="local", contextIndex=0)
        for _depth in range(5000):
            root = _node("Lean.Parser.Term.paren", root)

        target = compile_source_syntax_trace(_source_trace(root))

        self.assertEqual(target.node_count, 5000)
        self.assertEqual(target.local_reference_count, 1)


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

    def _save_raw_record(self, row_index: int, tactic: str) -> dict:
        row = DatasetRow(
            state="h : P\n⊢ P",
            target_state="no goals to be solved",
            theorem="Demo.result",
            tactic=tactic,
            split="train",
            row_index=row_index,
            dataset_name="fake/dataset",
            repo_url="https://example.invalid/mathlib4",
            repo_commit="commit",
            file_path="Mathlib/Demo.lean",
        )
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
        SExprCache(self.cache_root, project_path="").save("train", row_index, raw_record)
        return raw_record

    def _save_trace(
        self, cache_class, row_index: int, tactic: str, payload: dict
    ) -> None:
        raw_record = self._save_raw_record(row_index, tactic)
        cache_class(self.cache_root).save(
            "train",
            row_index,
            {
                "schema_version": cache_class.SCHEMA_VERSION,
                "extractor_version": cache_class.EXTRACTOR_VERSION,
                "raw_record_sha256": cache_class.raw_record_sha256(raw_record),
                "theorem": "Demo.result",
                "tactic": tactic,
                **payload,
            },
        )

    def test_audit_validates_digest_and_writes_targets(self) -> None:
        self._save_trace(
            ActionTraceCache,
            7,
            "exact h",
            {
                "terms": [{"source_start": 10, "action_sexp": "(:local FV3)"}],
                "syntax_args": [],
                "local_context": [{"context_index": 3, "user_name": "h"}],
            },
        )

        summary = run_action_target_audit(
            ActionTargetAuditConfig(
                cache_root=self.cache_root,
                output_dir=self.output_dir,
                trace_version="v2",
            )
        )

        train = summary["splits"]["train"]
        self.assertEqual(train["compiled_count"], 1)
        self.assertEqual(train["payload_row_count"], 1)
        self.assertIn('"op": "LOCAL"', (self.output_dir / "targets.jsonl").read_text())
        self.assertTrue((self.output_dir / "summary.md").exists())

    def test_audit_reads_source_syntax_traces_and_enforces_operation_cap(self) -> None:
        short = _node(
            "Lean.Parser.Tactic.exact",
            _atom("exact", 0),
            _identifier("h", 6, semanticRole="local", contextIndex=3),
        )
        long_root: dict = _identifier("h", 0, semanticRole="local", contextIndex=3)
        for _depth in range(20):
            long_root = _node("Lean.Parser.Term.paren", long_root)
        local_context = [{"context_index": 3, "user_name": "h"}]
        self._save_trace(
            SourceSyntaxTraceCache,
            7,
            "exact h",
            {
                "source_syntax": short,
                "syntax_args": [],
                "term_ranges": [],
                "local_context": local_context,
            },
        )
        self._save_trace(
            SourceSyntaxTraceCache,
            8,
            "exact ((((h))))",
            {
                "source_syntax": long_root,
                "syntax_args": [],
                "term_ranges": [],
                "local_context": local_context,
            },
        )

        summary = run_action_target_audit(
            ActionTargetAuditConfig(
                cache_root=self.cache_root,
                output_dir=self.output_dir,
                trace_version="v3",
                max_operations=10,
            )
        )

        train = summary["splits"]["train"]
        self.assertEqual(summary["trace_extractor_version"], "lean-action-trace-v3")
        self.assertEqual(train["compiled_count"], 2)
        self.assertEqual(train["accepted_count"], 1)
        self.assertEqual(train["over_cap_count"], 1)
        self.assertEqual(train["metric_totals"]["local_references"], 2)
        self.assertEqual(train["operation_counts"]["NODE"], 21)

        target_lines = (
            (self.output_dir / "targets.jsonl").read_text(encoding="utf-8").splitlines()
        )
        self.assertEqual(len(target_lines), 1)
        self.assertIn('"trace_version": "v3"', target_lines[0])
        self.assertIn(
            "target_too_long",
            (self.output_dir / "samples.jsonl").read_text(encoding="utf-8"),
        )

    def test_audit_ignores_the_other_trace_version(self) -> None:
        self._save_trace(
            ActionTraceCache,
            7,
            "exact h",
            {
                "terms": [{"source_start": 10, "action_sexp": "(:local FV3)"}],
                "syntax_args": [],
                "local_context": [{"context_index": 3, "user_name": "h"}],
            },
        )

        with self.assertRaisesRegex(FileNotFoundError, "action_trace_v3"):
            run_action_target_audit(
                ActionTargetAuditConfig(
                    cache_root=self.cache_root,
                    output_dir=self.output_dir,
                    trace_version="v3",
                )
            )


if __name__ == "__main__":
    unittest.main()
