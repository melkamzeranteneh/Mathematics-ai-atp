from __future__ import annotations

import json
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.preparation import SExprCache
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_inspection import (
    build_theorem_trace,
    theorem_trace_markdown,
    write_theorem_trace,
)


def _row(index: int, state: str, tactic: str, target: str) -> DatasetRow:
    return DatasetRow(
        state=state,
        theorem="Demo.theorem",
        tactic=tactic,
        target_state=target,
        split="train",
        row_index=index,
        repo_url="https://example.invalid/mathlib",
        repo_commit="abc123",
        file_path="Mathlib/Demo.lean",
    )


def _record(row: DatasetRow) -> dict[str, object]:
    return {
        "schema_version": SExprCache.SCHEMA_VERSION,
        "extractor_version": SExprCache.EXTRACTOR_VERSION,
        "dataset": row.dataset_name,
        "split": row.split,
        "row_index": row.row_index,
        "theorem": row.theorem,
        "repo_commit": row.repo_commit,
        "file_path": row.file_path,
        "state_sha256": SExprCache.row_state_sha256(row),
        "tactic_sha256": SExprCache.row_tactic_sha256(row),
        "target_state_sha256": SExprCache.row_target_state_sha256(row),
        "text_state": row.state,
        "text_target_state": row.target_state,
        "tactic": row.tactic,
        "goal_sexp": f"(:goal {row.row_index})",
        "hyp_sexps": [{"name": "x", "sexp": "Nat"}],
        "alignment_kind": "exact",
        "target_state_matches_invocation": True,
        "hypothesis_names_match": True,
        "pending_goal_count": 1,
        "unit_index": 0,
        "invocation_index": row.row_index,
    }


def test_builds_complete_ordered_trace(tmp_path: Path) -> None:
    first = _row(8, "x : Nat\n⊢ x = x", "rfl", "no goals")
    second = _row(3, "x : Nat\n⊢ True", "trivial", first.state)
    cache = SExprCache(tmp_path, project_path="")
    cache.save("train", first.row_index, _record(first))
    cache.save("train", second.row_index, _record(second))

    trace = build_theorem_trace(
        rows=[first, second],
        theorem="Demo.theorem",
        split="train",
        cache=cache,
    )

    assert trace["complete"] is True
    assert trace["dataset_row_count"] == 2
    assert [step["row_index"] for step in trace["steps"]] == [3, 8]
    assert trace["steps"][0]["transition_matches_next_dataset_row"] is True
    assert trace["steps"][1]["transition_matches_next_dataset_row"] is None


def test_reports_missing_or_stale_cache_rows(tmp_path: Path) -> None:
    valid = _row(1, "⊢ True", "trivial", "no goals")
    missing = _row(2, "⊢ False", "contradiction", "no goals")
    cache = SExprCache(tmp_path, project_path="")
    cache.save("train", valid.row_index, _record(valid))

    trace = build_theorem_trace(
        rows=[valid, missing],
        theorem="Demo.theorem",
        split="train",
        cache=cache,
    )

    assert trace["complete"] is False
    assert trace["cached_row_count"] == 1
    assert trace["missing_row_indices"] == [2]


def test_markdown_and_json_preserve_full_expressions(tmp_path: Path) -> None:
    row = _row(4, "x : Nat\n⊢ x = x", "exact rfl", "no goals")
    cache = SExprCache(tmp_path / "cache", project_path="")
    cache.save("train", row.row_index, _record(row))
    trace = build_theorem_trace(
        rows=[row],
        theorem="Demo.theorem",
        split="train",
        cache=cache,
    )

    markdown = theorem_trace_markdown(trace)
    assert "(:goal 4)" in markdown
    assert "#### `x`" in markdown
    assert "exact rfl" in markdown

    json_path, markdown_path = write_theorem_trace(
        trace, output_dir=tmp_path / "reports"
    )
    assert json.loads(json_path.read_text(encoding="utf-8"))["complete"] is True
    assert "(:goal 4)" in markdown_path.read_text(encoding="utf-8")
