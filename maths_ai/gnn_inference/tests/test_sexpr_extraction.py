from __future__ import annotations

import copy
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.preparation import (
    SExprCache,
    SExprUnavailableError,
    prepare_example,
)
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import (
    DATASET_MATHLIB_COMMIT,
    SExprExtractionConfig,
    extract_sexpressions,
    extract_split_with_client,
)


def _goal(target: str, *names: str) -> dict[str, object]:
    return {
        "target": {"pp": target, "sexp": f"((:c {target}) 0)"},
        "vars": [
            {
                "name": f"internal_{name}",
                "userName": name,
                "type": {"pp": "Prop", "sexp": "(:sort 0)"},
            }
            for name in names
        ],
    }


def _invocation(before: str, tactic: str, after: str, target: str) -> dict[str, object]:
    return {
        "goalBefore": before,
        "goalAfter": after,
        "tactic": tactic,
        "goalsBefore": [_goal(target, "p")],
        "goalsAfter": [],
    }


class _FakeClient:
    def __init__(self, units: list[dict[str, object]]) -> None:
        self.units = units
        self.files: list[str] = []
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True
        return self

    async def process_file(self, file_path: str):
        self.files.append(file_path)
        return self.units

    async def close(self):
        self.closed = True


class SExprExtractionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.root = Path("maths_ai/gnn_inference/tests/_tmp_sexpr_extraction")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.source_root = self.root / "mathlib"
        self.file_path = "Mathlib/Demo.lean"
        source_file = self.source_root / self.file_path
        source_file.parent.mkdir(parents=True)
        self.source = "theorem Demo.theorem (p : Prop) : First p := by\n  advance\n  finish\n"
        source_file.write_text(self.source, encoding="utf-8")
        self.cache = SExprCache(self.root / "prepared", str(self.source_root))
        self.rows = [
            DatasetRow(
                state="p : Prop\n⊢ First p",
                target_state="p : Prop\n⊢ Second p",
                theorem="Demo.theorem",
                tactic="advance",
                split="train",
                row_index=10,
                dataset_name="fake/dataset",
                repo_url="https://example.invalid/mathlib4",
                repo_commit=DATASET_MATHLIB_COMMIT,
                file_path=self.file_path,
            ),
            DatasetRow(
                state="p : Prop\n⊢ Second p",
                target_state="no goals to be solved",
                theorem="Demo.theorem",
                tactic="finish",
                split="train",
                row_index=11,
                dataset_name="fake/dataset",
                repo_url="https://example.invalid/mathlib4",
                repo_commit=DATASET_MATHLIB_COMMIT,
                file_path=self.file_path,
            ),
        ]

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def _units(self) -> list[dict[str, object]]:
        return [
            {
                "boundary": [0, len(self.source.encode("utf-8"))],
                "messages": [],
                "invocations": [
                    _invocation(self.rows[0].state, "advance", self.rows[0].target_state, "First"),
                    _invocation(self.rows[1].state, "finish", "", "Second"),
                ],
            }
        ]

    async def _extract(self, client: _FakeClient, rows=None, *, resume=True):
        return await extract_split_with_client(
            client,
            rows=self.rows if rows is None else rows,
            cache=self.cache,
            prepared_root=self.root / "prepared",
            source_root=self.source_root,
            split="train",
            resume=resume,
        )

    async def test_extracts_authentic_invocations_and_source_coordinates(self):
        client = _FakeClient(self._units())
        manifest = await self._extract(client)

        self.assertEqual(manifest["compiled_files"], 1)
        self.assertEqual(manifest["extracted_rows"], 2)
        self.assertEqual(manifest["coverage"], 1.0)
        self.assertEqual(client.files, [self.file_path])
        first = self.cache.load("train", 10)
        self.assertEqual(first["schema_version"], SExprCache.SCHEMA_VERSION)
        self.assertEqual(first["extractor_version"], "source-invocation-v4")
        self.assertEqual(first["repo_commit"], DATASET_MATHLIB_COMMIT)
        self.assertEqual(first["file_path"], self.file_path)
        self.assertEqual(first["goal_sexp"], "((:c First) 0)")
        self.assertEqual(first["hyp_sexps"], [{"name": "p", "sexp": "(:sort 0)"}])
        self.assertEqual(first["invocation_index"], 0)
        self.assertEqual(first["alignment_kind"], "exact")
        self.assertTrue(first["target_state_matches_invocation"])

    async def test_resume_skips_source_compilation_when_all_rows_are_valid(self):
        await self._extract(_FakeClient(self._units()))
        resumed = _FakeClient(self._units())
        manifest = await self._extract(resumed)

        self.assertEqual(manifest["cached_rows"], 2)
        self.assertEqual(manifest["compiled_files"], 0)
        self.assertEqual(resumed.files, [])

    async def test_target_state_change_invalidates_cache_and_alignment(self):
        await self._extract(_FakeClient(self._units()))
        changed = [
            self.rows[0],
            DatasetRow(**{**self.rows[1].__dict__, "target_state": "different goal"}),
        ]
        manifest = await self._extract(_FakeClient(self._units()), changed)

        self.assertEqual(manifest["failed_rows"], 1)
        self.assertEqual(manifest["failure_phases"], {"invocation_alignment": 1})

    async def test_dataset_links_and_qualified_names_match_lean_tactics(self):
        linked = DatasetRow(
            **{
                **self.rows[0].__dict__,
                "tactic": (
                    "apply <a>Namespace.Theorem</a> "
                    "(<a>Algebra.helper</a> p)"
                ),
            }
        )
        units = self._units()
        units[0]["invocations"][0]["tactic"] = "apply Theorem (helper p)"

        manifest = await self._extract(_FakeClient(units), [linked])

        self.assertEqual(manifest["coverage"], 1.0)

    async def test_semantically_identical_duplicate_invocations_are_collapsed(self):
        units = self._units()
        units[0]["invocations"].insert(
            1, copy.deepcopy(units[0]["invocations"][0])
        )
        manifest = await self._extract(_FakeClient(units), [self.rows[0]])

        self.assertEqual(manifest["coverage"], 1.0)
        self.assertIsNotNone(self.cache.load("train", 10))

    async def test_different_serialized_inputs_are_rejected_instead_of_guessed(self):
        units = self._units()
        duplicate = copy.deepcopy(units[0]["invocations"][0])
        duplicate["goalsBefore"][0]["target"]["sexp"] = "(:c Different)"
        units[0]["invocations"].insert(1, duplicate)
        manifest = await self._extract(_FakeClient(units), [self.rows[0]])

        self.assertEqual(manifest["failed_rows"], 1)
        self.assertEqual(manifest["failure_phases"], {"ambiguous_invocation": 1})
        self.assertIsNone(self.cache.load("train", 10))

    async def test_branch_opener_uses_authentic_input_goal(self):
        branch_row = DatasetRow(
            **{
                **self.rows[0].__dict__,
                "tactic": "by_cases h",
            }
        )
        units = self._units()
        invocation = units[0]["invocations"][0]
        invocation["tactic"] = "by_cases h\n· finish\n· finish\n"
        invocation["goalAfter"] = ""

        manifest = await self._extract(_FakeClient(units), [branch_row])

        self.assertEqual(manifest["coverage"], 1.0)
        record = self.cache.load("train", 10)
        self.assertEqual(record["goal_sexp"], "((:c First) 0)")
        self.assertEqual(record["alignment_kind"], "branch_opener")
        self.assertFalse(record["target_state_matches_invocation"])

    async def test_goal_stack_order_need_not_equal_source_tree_order(self):
        units = self._units()
        units[0]["invocations"].reverse()

        manifest = await self._extract(_FakeClient(units))

        self.assertEqual(manifest["coverage"], 1.0)
        self.assertIsNotNone(self.cache.load("train", 10))
        self.assertIsNotNone(self.cache.load("train", 11))

    async def test_capture_failure_commits_no_partial_theorem(self):
        units = self._units()
        units[0]["invocations"][1]["goalsBefore"][0]["target"].pop("sexp")
        manifest = await self._extract(_FakeClient(units))

        self.assertEqual(manifest["failed_rows"], 2)
        self.assertEqual(manifest["failure_phases"], {"sexpr_capture": 2})
        self.assertIsNone(self.cache.load("train", 10))
        self.assertIsNone(self.cache.load("train", 11))

    async def test_reported_pantograph_capture_error_commits_no_partial_theorem(self):
        units = self._units()
        units[0]["invocations"][1]["captureError"] = "synthetic serializer failure"
        manifest = await self._extract(_FakeClient(units))

        self.assertEqual(manifest["failed_rows"], 2)
        self.assertEqual(manifest["failure_phases"], {"sexpr_capture": 2})
        self.assertIsNone(self.cache.load("train", 10))
        self.assertIsNone(self.cache.load("train", 11))

    async def test_wrong_dataset_commit_is_rejected_before_compile(self):
        wrong = [DatasetRow(**{**self.rows[0].__dict__, "repo_commit": "wrong"})]
        client = _FakeClient(self._units())
        manifest = await self._extract(client, wrong)

        self.assertEqual(manifest["failure_phases"], {"commit_mismatch": 1})
        self.assertEqual(client.files, [])

    async def test_missing_source_metadata_is_reported(self):
        missing = [DatasetRow(**{**self.rows[0].__dict__, "file_path": ""})]
        manifest = await self._extract(_FakeClient(self._units()), missing)
        self.assertEqual(manifest["failure_phases"], {"source_metadata": 1})

    def test_legacy_cache_is_rejected(self):
        path = self.root / "prepared" / "train" / "sexpr" / "000000010.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": 2, "goal_sexp": "(:sort 0)"}), encoding="utf-8")
        self.assertIsNone(self.cache.load("train", 10))

    def test_strict_preparation_refuses_missing_cache(self):
        with self.assertRaises(SExprUnavailableError):
            prepare_example(self.rows[0], sexpr_cache=self.cache, use_sexpr=True)

    async def test_top_level_writes_summary_and_closes_client(self):
        client = _FakeClient(self._units())

        with patch(
            "maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction.iter_dataset_rows",
            return_value=iter(self.rows),
        ):
            summary = await extract_sexpressions(
                SExprExtractionConfig(
                    prepared_root=self.root / "prepared",
                    source_root=self.source_root,
                    pantograph_repl=self.root / "fake-repl",
                    dataset_name="fake/dataset",
                    splits=("train",),
                    verify_source_commit=False,
                ),
                client_factory=lambda _config: client,
            )

        self.assertTrue(client.started)
        self.assertTrue(client.closed)
        self.assertEqual(summary["coverage"], 1.0)
        self.assertIn(
            "100.0000%",
            (self.root / "prepared" / "sexpr_extraction" / "summary.md").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
