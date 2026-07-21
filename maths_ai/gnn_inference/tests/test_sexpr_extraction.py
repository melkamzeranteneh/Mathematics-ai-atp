from __future__ import annotations

import json
import shutil
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.preparation import (
    SExprCache,
    SExprUnavailableError,
    prepare_example,
)
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import (
    SExprExtractionConfig,
    extract_sexpressions,
    extract_split_with_server,
)


@dataclass
class _Variable:
    name: str
    t: str = "(:sort 0)"


@dataclass
class _Goal:
    target: str
    variables: list[_Variable]


@dataclass
class _GoalState:
    goals: list[_Goal]

    @property
    def is_solved(self) -> bool:
        return not self.goals


class _FakeServer:
    def __init__(
        self,
        states: list[_GoalState],
        *,
        fail_tactic: str | None = None,
    ) -> None:
        self.states = states
        self.fail_tactic = fail_tactic
        self.calls: list[tuple[str, str]] = []
        self._introduced: list[str] = []
        self._step = 0
        self.shutdown_called = False

    async def env_inspect_async(self, theorem: str):
        self.calls.append(("inspect", theorem))
        return {"type": {"pp": "∀ (p : Prop), Prop"}}

    async def goal_start_async(self, theorem_type: str):
        self.calls.append(("start", theorem_type))
        return _GoalState([_Goal("(:sort 0)", [])])

    async def goal_tactic_async(self, goal_state: _GoalState, tactic: str):
        self.calls.append(("tactic", tactic))
        if tactic.startswith("intro"):
            self._introduced.append(tactic.partition(" ")[2] or "_")
            first = self.states[0]
            return _GoalState(
                [
                    _Goal(
                        first.goals[0].target,
                        [_Variable(name) for name in self._introduced],
                    )
                ]
            )
        if tactic == self.fail_tactic:
            raise RuntimeError("configured tactic failure")
        self._step += 1
        return self.states[self._step] if self._step < len(self.states) else _GoalState([])

    async def shutdown_async(self):
        self.shutdown_called = True


class SExprExtractionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.root = Path("maths_ai/gnn_inference/tests/_tmp_sexpr_extraction")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.cache = SExprCache(self.root, "fake/project")
        self.rows = [
            DatasetRow(
                state="p : Prop\n⊢ First p",
                theorem="Demo.theorem",
                tactic="advance",
                split="train",
                row_index=10,
                dataset_name="fake/dataset",
            ),
            DatasetRow(
                state="p : Prop\n⊢ Second p",
                theorem="Demo.theorem",
                tactic="finish",
                split="train",
                row_index=11,
                dataset_name="fake/dataset",
            ),
        ]

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def _server(self, *, fail_tactic: str | None = None):
        return _FakeServer(
            [
                _GoalState([_Goal("((:c First) 0)", [_Variable("p")])]),
                _GoalState([_Goal("((:c Second) 0)", [_Variable("p")])]),
            ],
            fail_tactic=fail_tactic,
        )

    async def _extract(self, server, *, resume: bool = True):
        return await extract_split_with_server(
            server,
            rows=self.rows,
            cache=self.cache,
            prepared_root=self.root,
            split="train",
            resume=resume,
        )

    async def test_replays_theorem_sequentially_and_writes_versioned_rows(self):
        server = self._server()
        manifest = await self._extract(server)

        self.assertEqual(manifest["extracted_theorems"], 1)
        self.assertEqual(manifest["extracted_rows"], 2)
        self.assertEqual(manifest["coverage"], 1.0)
        self.assertEqual(
            [call for call in server.calls if call[0] == "tactic"],
            [("tactic", "intro p"), ("tactic", "advance"), ("tactic", "finish")],
        )

        first = self.cache.load("train", 10)
        second = self.cache.load("train", 11)
        self.assertEqual(first["schema_version"], SExprCache.SCHEMA_VERSION)
        self.assertEqual(first["step_index"], 0)
        self.assertEqual(second["step_index"], 1)
        self.assertEqual(first["goal_sexp"], "((:c First) 0)")
        self.assertEqual(second["goal_sexp"], "((:c Second) 0)")
        self.assertNotEqual(first["state_sha256"], second["state_sha256"])

    async def test_resume_skips_only_a_complete_valid_theorem(self):
        await self._extract(self._server())
        resumed_server = self._server()

        manifest = await self._extract(resumed_server)

        self.assertEqual(manifest["cached_theorems"], 1)
        self.assertEqual(manifest["cached_rows"], 2)
        self.assertEqual(resumed_server.calls, [])

    async def test_partial_cache_forces_replay_from_theorem_start(self):
        await self._extract(self._server())
        (self.root / "train" / "sexpr" / "000000011.json").unlink()
        replay_server = self._server()

        manifest = await self._extract(replay_server)

        self.assertEqual(manifest["extracted_theorems"], 1)
        self.assertIn(("inspect", "Demo.theorem"), replay_server.calls)
        self.assertIsNotNone(self.cache.load("train", 11))

    async def test_changed_dataset_tactic_invalidates_theorem_cache(self):
        await self._extract(self._server())
        changed_rows = [
            self.rows[0],
            DatasetRow(
                state=self.rows[1].state,
                theorem=self.rows[1].theorem,
                tactic="new_finish",
                split=self.rows[1].split,
                row_index=self.rows[1].row_index,
                dataset_name=self.rows[1].dataset_name,
            ),
        ]
        server = self._server()
        manifest = await extract_split_with_server(
            server,
            rows=changed_rows,
            cache=self.cache,
            prepared_root=self.root,
            split="train",
        )

        self.assertEqual(manifest["extracted_theorems"], 1)
        self.assertIn(("inspect", "Demo.theorem"), server.calls)
        self.assertEqual(self.cache.load("train", 11)["tactic"], "new_finish")

    async def test_alignment_failure_writes_report_and_no_partial_cache(self):
        # The first state aligns, but the replayed second state has only p.
        bad_rows = [
            self.rows[0],
            DatasetRow(
                state="p q : Prop\n⊢ Second p",
                theorem=self.rows[1].theorem,
                tactic=self.rows[1].tactic,
                split="train",
                row_index=11,
                dataset_name="fake/dataset",
            ),
        ]
        server = self._server()
        manifest = await extract_split_with_server(
            server,
            rows=bad_rows,
            cache=self.cache,
            prepared_root=self.root,
            split="train",
        )

        self.assertEqual(manifest["failed_theorems"], 1)
        self.assertEqual(manifest["failure_phases"], {"state_alignment": 1})
        self.assertIsNone(self.cache.load("train", 10))
        failure = json.loads(
            (self.root / "sexpr_extraction" / "failures" / "train.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        self.assertEqual(failure["phase"], "state_alignment")

    async def test_tactic_failure_commits_no_rows_from_theorem(self):
        manifest = await self._extract(self._server(fail_tactic="finish"))

        self.assertEqual(manifest["failed_theorems"], 1)
        self.assertEqual(manifest["failure_phases"], {"tactic_replay": 1})
        self.assertIsNone(self.cache.load("train", 10))
        self.assertIsNone(self.cache.load("train", 11))

    def test_legacy_unversioned_cache_is_rejected(self):
        path = self.root / "train" / "sexpr" / "000000010.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"goal_sexp": "(:sort 0)", "hyp_sexps": []}),
            encoding="utf-8",
        )

        self.assertIsNone(self.cache.load("train", 10))

    def test_strict_preparation_refuses_missing_cache(self):
        with self.assertRaises(SExprUnavailableError):
            prepare_example(self.rows[0], sexpr_cache=self.cache, use_sexpr=True)

    async def test_top_level_extraction_writes_summary_and_closes_server(self):
        server = self._server()

        async def factory(**kwargs):
            self.assertEqual(kwargs["options"], {"printExprAST": True})
            self.assertEqual(kwargs["timeout"], 600)
            return server

        with patch(
            "maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction.iter_dataset_rows",
            return_value=iter(self.rows),
        ):
            summary = await extract_sexpressions(
                SExprExtractionConfig(
                    prepared_root=self.root,
                    dataset_name="fake/dataset",
                    splits=("train",),
                    project_path="fake/project",
                ),
                server_factory=factory,
            )

        self.assertEqual(summary["coverage"], 1.0)
        self.assertTrue(server.shutdown_called)
        self.assertTrue((self.root / "sexpr_extraction" / "summary.json").exists())
        self.assertIn(
            "100.0000%",
            (self.root / "sexpr_extraction" / "summary.md").read_text(
                encoding="utf-8"
            ),
        )


if __name__ == "__main__":
    unittest.main()
