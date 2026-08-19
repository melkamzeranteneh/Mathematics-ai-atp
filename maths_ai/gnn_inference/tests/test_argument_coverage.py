from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from maths_ai.gnn_inference.atp_lean_gnn.action_targets import ActionTargetError
from maths_ai.gnn_inference.atp_lean_gnn.argument_coverage import (
    ArgumentCoverageConfig,
    classify_argument_slots,
    classify_structured_argument_slots,
    hypothesis_names_by_context_index,
    run_argument_coverage_audit,
)
from maths_ai.gnn_inference.atp_lean_gnn.cache import write_pyg_artifact
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.preparation import (
    SExprCache,
    SourceSyntaxTraceCache,
)


def _example(
    tactic_raw: str,
    tactic_name: str,
    *,
    node_indices: list[int],
    premise_mask: list[bool],
    lemma_ids: list[int] | None = None,
) -> Data:
    data = Data(
        x=torch.arange(len(premise_mask), dtype=torch.long),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    data.tactic_raw = tactic_raw
    data.tactic_name = tactic_name
    data.arg_node_indices = torch.tensor(node_indices, dtype=torch.long)
    data.arg_lemma_ids = torch.tensor(lemma_ids or [], dtype=torch.long)
    data.arg_count = len(node_indices)
    data.premise_mask = torch.tensor(premise_mask, dtype=torch.bool)
    data.row_index = 17
    data.theorem = "Demo.theorem"
    return data


def _graph(label_ids: list[int], premise_mask: list[bool]) -> Data:
    data = Data(
        x=torch.tensor(label_ids, dtype=torch.long),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    data.premise_mask = torch.tensor(premise_mask, dtype=torch.bool)
    data.row_index = 17
    data.theorem = "Demo.theorem"
    return data


def _identifier(source: str, **annotations) -> dict:
    return {
        "tag": "identifier",
        "source": source,
        "sourceStart": 0,
        "sourceEnd": len(source),
        **annotations,
    }


def _structured_trace(*children: dict, local_context=((0, "h"),)) -> dict:
    return {
        "tactic": "exact h",
        "source_syntax": {
            "tag": "node",
            "kind": "Lean.Parser.Tactic.exact",
            "source": "",
            "children": [
                {"tag": "atom", "source": "exact", "sourceStart": 0, "sourceEnd": 5},
                *children,
            ],
        },
        "syntax_args": [],
        "local_context": [
            {"context_index": index, "user_name": name}
            for index, name in local_context
        ],
    }


class ArgumentCoverageClassificationTests(unittest.TestCase):
    def test_classifies_selectable_and_masked_local_targets(self) -> None:
        selectable, _ = classify_argument_slots(
            _example(
                "apply h",
                "apply",
                node_indices=[1],
                premise_mask=[False, True],
            ),
            split="val",
        )
        masked, _ = classify_argument_slots(
            _example(
                "apply h",
                "apply",
                node_indices=[1],
                premise_mask=[False, False],
            ),
            split="val",
        )

        self.assertEqual(selectable[0]["category"], "local_selectable")
        self.assertEqual(masked[0]["category"], "local_present_but_masked")

    def test_recovers_global_lemma_from_cached_id_or_corpus(self) -> None:
        cached, _ = classify_argument_slots(
            _example(
                "exact Mathlib.result",
                "exact",
                node_indices=[-1],
                lemma_ids=[42],
                premise_mask=[False],
            ),
            split="test",
        )
        recovered, _ = classify_argument_slots(
            _example(
                "exact Mathlib.result",
                "exact",
                node_indices=[-1],
                premise_mask=[False],
            ),
            split="test",
            lemma_names={"Mathlib.result"},
        )

        self.assertEqual(cached[0]["category"], "global_library_lemma")
        self.assertEqual(recovered[0]["category"], "global_library_lemma")

    def test_classifies_missing_fresh_literal_and_qualified_targets(self) -> None:
        cases = [
            ("apply", "apply", "parser_missing_argument"),
            ("intro h", "intro", "fresh_identifier"),
            ("exact 42", "exact", "literal"),
            ("apply Foo.bar", "apply", "unresolved_qualified_name"),
        ]
        for tactic_raw, tactic_name, expected in cases:
            with self.subTest(tactic=tactic_raw):
                records, _ = classify_argument_slots(
                    _example(
                        tactic_raw,
                        tactic_name,
                        node_indices=[-1],
                        premise_mask=[False],
                    ),
                    split="train",
                )
                self.assertEqual(records[0]["category"], expected)

    def test_marks_excess_parser_tokens_as_semantically_ambiguous(self) -> None:
        records, row = classify_argument_slots(
            _example(
                "exact f x",
                "exact",
                node_indices=[0, -1],
                premise_mask=[True],
            ),
            split="val",
        )

        self.assertEqual(row["expected_arity"], 1)
        self.assertEqual(row["parsed_argument_count"], 2)
        self.assertEqual(row["ignored_parsed_token_count"], 1)
        self.assertEqual(row["parse_shape"], "excess_tokens_or_compound_expression")
        self.assertTrue(records[0]["semantically_ambiguous_parse"])

    def test_only_first_let_slot_is_a_fresh_identifier(self) -> None:
        records, _ = classify_argument_slots(
            _example(
                "let result value",
                "let",
                node_indices=[-1, -1],
                premise_mask=[False],
            ),
            split="train",
        )

        self.assertEqual(
            [record["category"] for record in records],
            ["fresh_identifier", "unmapped_token"],
        )

    def test_rejects_two_lemma_name_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "either lemma_corpus or lemma_index"):
            ArgumentCoverageConfig(
                prepared_root=Path("prepared"),
                output_dir=Path("output"),
                lemma_corpus=Path("lemmas.jsonl"),
                lemma_index=Path("index"),
            ).normalized()


class StructuredArgumentCoverageTests(unittest.TestCase):
    """Classify Lean-annotated syntax targets against the graph that will be pointed at."""

    NODE_VOCAB = {"<UNK>": 0, "State": 1, "Hyp": 2, "FV0": 3, "FV1": 4}

    def test_local_reference_resolves_to_a_selectable_graph_node(self) -> None:
        # Node 2 carries the label id of ``FV0``, and the premise mask marks it.
        records, row = classify_structured_argument_slots(
            _graph([1, 2, 3], [False, False, True]),
            _structured_trace(_identifier("h", semanticRole="local", contextIndex=0)),
            split="val",
            node_vocab=self.NODE_VOCAB,
            hypothesis_names={0: "h"},
        )

        self.assertEqual([record["category"] for record in records], ["local_selectable"])
        self.assertEqual(records[0]["node_id"], 2)
        self.assertEqual(row["reference_slot_count"], 1)
        self.assertEqual(row["local_reference_count"], 1)
        self.assertTrue(row["resolved"])

    def test_masked_and_absent_local_nodes_are_reported_separately(self) -> None:
        masked, _ = classify_structured_argument_slots(
            _graph([1, 2, 3], [False, False, False]),
            _structured_trace(_identifier("h", semanticRole="local", contextIndex=0)),
            split="val",
            node_vocab=self.NODE_VOCAB,
            hypothesis_names={0: "h"},
        )
        absent, _ = classify_structured_argument_slots(
            _graph([1, 2], [False, False]),
            _structured_trace(_identifier("h", semanticRole="local", contextIndex=0)),
            split="val",
            node_vocab=self.NODE_VOCAB,
            hypothesis_names={0: "h"},
        )
        outside_vocab, _ = classify_structured_argument_slots(
            _graph([1, 2, 3], [False, False, True]),
            _structured_trace(
                _identifier("h", semanticRole="local", contextIndex=1),
                local_context=((1, "h"),),
            ),
            split="val",
            node_vocab={"<UNK>": 0, "State": 1},
            hypothesis_names={1: "h"},
        )

        self.assertEqual(masked[0]["category"], "local_present_but_masked")
        self.assertEqual(absent[0]["category"], "local_node_absent")
        self.assertEqual(outside_vocab[0]["category"], "local_label_outside_vocab")

    def test_context_index_disagreement_is_flagged_instead_of_masked_verdict(self) -> None:
        records, row = classify_structured_argument_slots(
            _graph([1, 2, 3], [False, False, True]),
            _structured_trace(_identifier("h", semanticRole="local", contextIndex=0)),
            split="val",
            node_vocab=self.NODE_VOCAB,
            hypothesis_names={0: "different"},
        )
        absent_from_state, _ = classify_structured_argument_slots(
            _graph([1, 2, 3], [False, False, True]),
            _structured_trace(_identifier("h", semanticRole="local", contextIndex=0)),
            split="val",
            node_vocab=self.NODE_VOCAB,
            hypothesis_names={4: "h"},
        )

        self.assertEqual(records[0]["category"], "local_name_mismatch")
        self.assertFalse(row["resolved"])
        self.assertEqual(absent_from_state[0]["category"], "local_absent_from_state")

    def test_inaccessible_names_do_not_count_as_disagreement(self) -> None:
        records, _ = classify_structured_argument_slots(
            _graph([1, 2, 3], [False, False, True]),
            _structured_trace(
                _identifier("h", semanticRole="local", contextIndex=0),
                local_context=((0, "h✝"),),
            ),
            split="val",
            node_vocab=self.NODE_VOCAB,
            hypothesis_names={0: "h"},
        )

        self.assertEqual(records[0]["category"], "local_selectable")

    def test_global_reference_is_checked_against_the_lemma_corpus(self) -> None:
        known, _ = classify_structured_argument_slots(
            _graph([1], [False]),
            _structured_trace(
                _identifier("Demo.f", semanticRole="global", name="Demo.f")
            ),
            split="test",
            node_vocab=self.NODE_VOCAB,
            lemma_names={"Demo.f"},
        )
        unknown, _ = classify_structured_argument_slots(
            _graph([1], [False]),
            _structured_trace(
                _identifier("Demo.f", semanticRole="global", name="Demo.f")
            ),
            split="test",
            node_vocab=self.NODE_VOCAB,
            lemma_names=set(),
        )
        unchecked, row = classify_structured_argument_slots(
            _graph([1], [False]),
            _structured_trace(
                _identifier("And.intro", semanticRole="constructor", name="And.intro")
            ),
            split="test",
            node_vocab=self.NODE_VOCAB,
        )

        self.assertEqual(known[0]["category"], "global_library_lemma")
        self.assertEqual(unknown[0]["category"], "global_outside_corpus")
        self.assertEqual(unchecked[0]["category"], "global_unchecked")
        self.assertEqual(row["local_reference_count"], 0)

    def test_scoped_and_unannotated_identifiers_are_distinct_positions(self) -> None:
        records, row = classify_structured_argument_slots(
            _graph([1], [False]),
            _structured_trace(
                _identifier("x", semanticRole="scoped_local"),
                _identifier("foo"),
                {"tag": "missing"},
            ),
            split="train",
            node_vocab=self.NODE_VOCAB,
        )

        self.assertEqual(
            [record["category"] for record in records],
            ["tactic_scoped_binding", "unannotated_identifier", "missing_syntax"],
        )
        self.assertEqual(row["reference_slot_count"], 3)
        self.assertFalse(row["resolved"])

    def test_structural_operations_are_not_counted_as_naming_positions(self) -> None:
        records, row = classify_structured_argument_slots(
            _graph([1, 2, 3], [False, False, True]),
            _structured_trace(_identifier("h", semanticRole="local", contextIndex=0)),
            split="train",
            node_vocab=self.NODE_VOCAB,
        )

        # One node, one atom, one identifier leaf, plus the trailing STOP.
        self.assertEqual(row["operation_count"], 4)
        self.assertEqual(row["structural_operation_count"], 2)
        self.assertEqual(len(records), 1)

    def test_unknown_context_index_fails_compilation(self) -> None:
        with self.assertRaises(ActionTargetError) as caught:
            classify_structured_argument_slots(
                _graph([1], [False]),
                _structured_trace(
                    _identifier("h", semanticRole="local", contextIndex=9)
                ),
                split="train",
                node_vocab=self.NODE_VOCAB,
            )

        self.assertEqual(caught.exception.code, "unknown_local_reference")

    def test_hypothesis_names_come_from_the_validated_raw_record(self) -> None:
        names = hypothesis_names_by_context_index(
            {
                "hyp_sexps": [
                    {"name": "h", "context_index": 0, "sexp": "(:c P)"},
                    {"name": "hx", "context_index": 3, "sexp": "(:c Q)"},
                    {"name": "text-only", "sexp": "(:c R)"},
                    "malformed",
                ]
            }
        )

        self.assertEqual(names, {0: "h", 3: "hx"})

    def test_records_without_context_indices_disable_the_cross_check(self) -> None:
        # Raw source-faithful records carry no context index.  Reading their
        # absence as "index not in the state" would report every local
        # reference as a skew that does not exist.
        self.assertIsNone(
            hypothesis_names_by_context_index(
                {"hyp_sexps": [{"name": "h", "sexp": "(:c P)"}]}
            )
        )

        records, _ = classify_structured_argument_slots(
            _graph([1, 2, 3], [False, False, True]),
            _structured_trace(
                _identifier("h", semanticRole="local", contextIndex=0)
            ),
            split="train",
            node_vocab=self.NODE_VOCAB,
            hypothesis_names=hypothesis_names_by_context_index(
                {"hyp_sexps": [{"name": "h", "sexp": "(:c P)"}]}
            ),
        )

        self.assertEqual(records[0]["category"], "local_selectable")


class StructuredArgumentCoverageAuditTests(unittest.TestCase):
    """Run both metrics over one prepared root containing version-3 sidecars."""

    NODE_VOCAB = {"<UNK>": 0, "State": 1, "Hyp": 2, "FV0": 3}

    def setUp(self) -> None:
        self.root = Path("maths_ai/gnn_inference/tests/_tmp_argument_coverage")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.prepared_root = self.root / "prepared"
        self.output_dir = self.root / "report"
        self._prepare_corpus_root(self.prepared_root, node_vocab=self.NODE_VOCAB)

    def _prepare_corpus_root(self, root: Path, *, node_vocab: dict[str, int]) -> None:
        # Only the audited split is created here.  An audit must not demand the
        # sibling manifests it never reads: a partially rebuilt corpus has none,
        # and fabricating empty ones to satisfy the loader would also hide a
        # genuinely missing artifact directory.
        (root / "vocab").mkdir(parents=True)
        (root / "manifests").mkdir(parents=True)
        (root / "vocab" / "node_vocab.json").write_text(
            json.dumps(node_vocab), encoding="utf-8"
        )
        (root / "vocab" / "tactic_vocab.json").write_text(
            json.dumps({"<UNK_TACTIC>": 0, "exact": 1}), encoding="utf-8"
        )
        (root / "train" / "pyg").mkdir(parents=True)
        (root / "manifests" / "train.json").write_text(
            json.dumps({"artifact_paths": {"pyg_dir": "train/pyg"}}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def _save_graph(self, row_index: int, *, root: Path | None = None) -> None:
        data = Data(
            x=torch.tensor([1, 2, 3], dtype=torch.long),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        data.premise_mask = torch.tensor([False, False, True], dtype=torch.bool)
        data.tactic_raw = "exact h"
        data.tactic_name = "exact"
        data.arg_node_indices = torch.tensor([2], dtype=torch.long)
        data.arg_lemma_ids = torch.tensor([], dtype=torch.long)
        data.row_index = row_index
        data.theorem = "Demo.result"
        write_pyg_artifact(
            root if root is not None else self.prepared_root,
            split="train",
            row_index=row_index,
            data=data,
        )

    def _save_raw_record(self, row_index: int) -> dict:
        row = DatasetRow(
            state="h : P\n⊢ P",
            target_state="no goals to be solved",
            theorem="Demo.result",
            tactic="exact h",
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
            "hyp_sexps": [{"name": "h", "context_index": 0, "sexp": "(:c P)"}],
        }
        SExprCache(self.prepared_root, project_path="").save(
            "train", row_index, raw_record
        )
        return raw_record

    def _save_structured_trace(
        self, row_index: int, *, extra_children: tuple[dict, ...] = ()
    ) -> None:
        raw_record = self._save_raw_record(row_index)
        SourceSyntaxTraceCache(self.prepared_root).save(
            "train",
            row_index,
            {
                "schema_version": SourceSyntaxTraceCache.SCHEMA_VERSION,
                "extractor_version": SourceSyntaxTraceCache.EXTRACTOR_VERSION,
                "raw_record_sha256": SourceSyntaxTraceCache.raw_record_sha256(
                    raw_record
                ),
                "theorem": "Demo.result",
                "tactic": "exact h",
                "source_syntax": {
                    "tag": "node",
                    "kind": "Lean.Parser.Tactic.exact",
                    "source": "",
                    "children": [
                        {
                            "tag": "atom",
                            "source": "exact",
                            "sourceStart": 0,
                            "sourceEnd": 5,
                        },
                        {
                            "tag": "identifier",
                            "source": "h",
                            "sourceStart": 6,
                            "sourceEnd": 7,
                            "semanticRole": "local",
                            "contextIndex": 0,
                        },
                        *extra_children,
                    ],
                },
                "syntax_args": [],
                "local_context": [{"context_index": 0, "user_name": "h"}],
            },
        )

    def test_audit_reports_both_metrics_over_the_traced_rows_only(self) -> None:
        self._save_graph(7)
        self._save_structured_trace(7)
        self._save_graph(8)

        summary = run_argument_coverage_audit(
            ArgumentCoverageConfig(
                prepared_root=self.prepared_root,
                output_dir=self.output_dir,
                splits=("train",),
                structured_traces=True,
            )
        )

        train = summary["splits"]["train"]
        structured = train["structured"]
        self.assertTrue(summary["structured_traces"])
        self.assertEqual(summary["context_index_node_labels"], 1)
        self.assertEqual(train["row_count"], 1)
        self.assertEqual(structured["trace_row_count"], 1)
        self.assertEqual(structured["rows_outside_trace_population"], 1)
        self.assertEqual(structured["row_count"], 1)
        self.assertEqual(structured["trace_states"], {"classified": 1})
        self.assertEqual(structured["categories"], {"local_selectable": 1})
        self.assertEqual(structured["local_selectable_coverage"], 1.0)
        self.assertEqual(structured["resolved_reference_coverage"], 1.0)
        self.assertEqual(structured["fully_resolved_row_count"], 1)
        self.assertEqual(
            structured["trace_extractor_version"],
            SourceSyntaxTraceCache.EXTRACTOR_VERSION,
        )

        report = (self.output_dir / "summary.md").read_text(encoding="utf-8")
        self.assertIn("### Lean-annotated syntax targets", report)
        self.assertIn("Local selectable coverage", report)
        samples = (
            (self.output_dir / "argument_samples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        metrics = {json.loads(line)["metric"] for line in samples}
        self.assertEqual(metrics, {"regex", "structured"})

    def test_corpus_without_context_index_nodes_says_so_in_the_report(self) -> None:
        # A corpus prepared from the raw S-expression variant carries no
        # FV{context_index} nodes at all, so every local reference is
        # unresolvable for a reason that has nothing to do with the mask.
        (self.prepared_root / "vocab" / "node_vocab.json").write_text(
            json.dumps({"<UNK>": 0, "State": 1, "Hyp": 2, "h": 3}), encoding="utf-8"
        )
        self._save_graph(7)
        self._save_structured_trace(7)

        summary = run_argument_coverage_audit(
            ArgumentCoverageConfig(
                prepared_root=self.prepared_root,
                output_dir=self.output_dir,
                splits=("train",),
                structured_traces=True,
            )
        )

        structured = summary["splits"]["train"]["structured"]
        self.assertEqual(summary["context_index_node_labels"], 0)
        self.assertEqual(structured["categories"], {"local_label_outside_vocab": 1})
        self.assertEqual(structured["local_selectable_coverage"], 0.0)

        report = (self.output_dir / "summary.md").read_text(encoding="utf-8")
        self.assertIn("--sexpr-variant model", report)

    def test_audit_reads_one_split_without_the_sibling_manifests(self) -> None:
        self._save_graph(7)
        self._save_structured_trace(7)
        self.assertFalse((self.prepared_root / "manifests" / "val.json").exists())
        self.assertFalse((self.prepared_root / "manifests" / "test.json").exists())

        summary = run_argument_coverage_audit(
            ArgumentCoverageConfig(
                prepared_root=self.prepared_root,
                output_dir=self.output_dir,
                splits=("train",),
                structured_traces=True,
            )
        )

        self.assertEqual(summary["splits"]["train"]["structured"]["row_count"], 1)

    def test_sidecars_may_live_outside_the_audited_prepared_root(self) -> None:
        # Rebuilding a corpus with a different representation produces a new
        # prepared root, while the costly S-expression and trace sidecars stay
        # where they were extracted.  The audit has to read across the two roots
        # without copies or symlinks standing in for the missing option.
        rebuilt_root = self.root / "rebuilt"
        self._prepare_corpus_root(rebuilt_root, node_vocab=self.NODE_VOCAB)
        self._save_graph(7, root=rebuilt_root)
        self._save_structured_trace(7)

        summary = run_argument_coverage_audit(
            ArgumentCoverageConfig(
                prepared_root=rebuilt_root,
                output_dir=self.output_dir,
                splits=("train",),
                structured_traces=True,
                sexpr_cache_root=self.prepared_root,
            )
        )

        structured = summary["splits"]["train"]["structured"]
        self.assertEqual(
            summary["sexpr_cache_root"], str(self.prepared_root.resolve())
        )
        self.assertEqual(structured["categories"], {"local_selectable": 1})

        with self.assertRaisesRegex(FileNotFoundError, "action_trace_v3"):
            run_argument_coverage_audit(
                ArgumentCoverageConfig(
                    prepared_root=rebuilt_root,
                    output_dir=self.output_dir,
                    splits=("train",),
                    structured_traces=True,
                    force=True,
                )
            )

    def test_unchecked_globals_are_excluded_from_resolved_coverage(self) -> None:
        self._save_graph(7)
        self._save_structured_trace(
            7,
            extra_children=(
                {
                    "tag": "identifier",
                    "source": "Demo.f",
                    "sourceStart": 8,
                    "sourceEnd": 14,
                    "semanticRole": "global",
                    "name": "Demo.f",
                },
            ),
        )
        corpus_dir = self.root / "lemmas"
        corpus_dir.mkdir(parents=True)
        (corpus_dir / "lemmas.jsonl").write_text(
            json.dumps(
                {
                    "lemma_id": 1,
                    "name": "Demo.f",
                    "statement": "True",
                    "namespace": "Demo",
                    "module": "Demo",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        unchecked = run_argument_coverage_audit(
            ArgumentCoverageConfig(
                prepared_root=self.prepared_root,
                output_dir=self.output_dir,
                splits=("train",),
                structured_traces=True,
            )
        )["splits"]["train"]["structured"]
        checked = run_argument_coverage_audit(
            ArgumentCoverageConfig(
                prepared_root=self.prepared_root,
                output_dir=self.root / "report_checked",
                splits=("train",),
                structured_traces=True,
                lemma_corpus=corpus_dir,
            )
        )["splits"]["train"]["structured"]

        # Without a candidate pool the global reference is unverifiable, so the
        # row is not fully resolved and only the local slot counts as resolved.
        self.assertEqual(
            unchecked["categories"], {"global_unchecked": 1, "local_selectable": 1}
        )
        self.assertEqual(unchecked["resolved_reference_coverage"], 0.5)
        self.assertEqual(unchecked["fully_resolved_row_count"], 0)
        self.assertIn(
            "excluded from the resolved coverage",
            (self.output_dir / "summary.md").read_text(encoding="utf-8"),
        )

        self.assertEqual(
            checked["categories"], {"global_library_lemma": 1, "local_selectable": 1}
        )
        self.assertEqual(checked["resolved_reference_coverage"], 1.0)
        self.assertEqual(checked["fully_resolved_row_count"], 1)

    def test_audit_requires_structured_sidecars_when_asked_for_them(self) -> None:
        self._save_graph(7)

        with self.assertRaisesRegex(FileNotFoundError, "action_trace_v3"):
            run_argument_coverage_audit(
                ArgumentCoverageConfig(
                    prepared_root=self.prepared_root,
                    output_dir=self.output_dir,
                    splits=("train",),
                    structured_traces=True,
                )
            )

    def test_stale_structured_sidecar_is_reported_rather_than_counted(self) -> None:
        self._save_graph(7)
        self._save_structured_trace(7)
        SourceSyntaxTraceCache(self.prepared_root).save(
            "train",
            7,
            {
                "schema_version": SourceSyntaxTraceCache.SCHEMA_VERSION,
                "extractor_version": SourceSyntaxTraceCache.EXTRACTOR_VERSION,
                "raw_record_sha256": "0" * 64,
                "theorem": "Demo.result",
                "tactic": "exact h",
                "source_syntax": {
                    "tag": "node",
                    "kind": "Lean.Parser.Tactic.exact",
                    "source": "",
                    "children": [],
                },
                "syntax_args": [],
                "local_context": [],
            },
        )

        summary = run_argument_coverage_audit(
            ArgumentCoverageConfig(
                prepared_root=self.prepared_root,
                output_dir=self.output_dir,
                splits=("train",),
                structured_traces=True,
            )
        )

        structured = summary["splits"]["train"]["structured"]
        self.assertEqual(structured["trace_states"], {"stale_or_unreadable_trace": 1})
        self.assertEqual(structured["row_count"], 0)
        self.assertEqual(structured["reference_slots"], 0)


if __name__ == "__main__":
    unittest.main()
