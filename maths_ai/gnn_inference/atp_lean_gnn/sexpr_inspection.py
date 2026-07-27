"""Human-readable inspection of cached theorem-level S-expression traces."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .dataset import DatasetRow
from .preparation import SExprCache


def _normalized_state(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())


def _fence(text: str, language: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text}\n{fence}"


def safe_theorem_filename(theorem: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", theorem).strip("._")
    return value or "theorem"


def build_theorem_trace(
    *,
    rows: Iterable[DatasetRow],
    theorem: str,
    split: str,
    cache: SExprCache,
) -> dict[str, object]:
    """Build a strictly validated trace for every dataset row of ``theorem``."""
    theorem_rows = sorted(
        (row for row in rows if row.theorem == theorem),
        key=lambda row: row.row_index,
    )
    if not theorem_rows:
        raise ValueError(f"Theorem '{theorem}' was not found in split '{split}'.")

    steps: list[dict[str, object]] = []
    missing_row_indices: list[int] = []
    for position, row in enumerate(theorem_rows):
        record = cache.load_for_row(
            row,
            extractor_version=SExprCache.EXTRACTOR_VERSION,
        )
        if record is None:
            missing_row_indices.append(row.row_index)
            continue

        next_row = theorem_rows[position + 1] if position + 1 < len(theorem_rows) else None
        transition_matches = (
            _normalized_state(row.target_state) == _normalized_state(next_row.state)
            if next_row is not None
            else None
        )
        steps.append(
            {
                "step": position + 1,
                "row_index": row.row_index,
                "tactic": row.tactic,
                "dataset_state": row.state,
                "dataset_target_state": row.target_state,
                "captured_state": record.get("text_state", ""),
                "captured_target_state": record.get("text_target_state", ""),
                "goal_sexp": record.get("goal_sexp", ""),
                "hyp_sexps": record.get("hyp_sexps", []),
                "alignment_kind": record.get("alignment_kind", ""),
                "target_state_matches_invocation": bool(
                    record.get("target_state_matches_invocation", False)
                ),
                "hypothesis_names_match": bool(
                    record.get("hypothesis_names_match", False)
                ),
                "pending_goal_count": record.get("pending_goal_count"),
                "transition_matches_next_dataset_row": transition_matches,
                "unit_index": record.get("unit_index"),
                "invocation_index": record.get("invocation_index"),
            }
        )

    return {
        "schema_version": 1,
        "dataset": theorem_rows[0].dataset_name,
        "split": split,
        "theorem": theorem,
        "repo_url": theorem_rows[0].repo_url,
        "repo_commit": theorem_rows[0].repo_commit,
        "file_path": theorem_rows[0].file_path,
        "dataset_row_count": len(theorem_rows),
        "cached_row_count": len(steps),
        "complete": not missing_row_indices,
        "missing_row_indices": missing_row_indices,
        "steps": steps,
    }


def theorem_trace_markdown(trace: dict[str, object]) -> str:
    steps = trace["steps"]
    assert isinstance(steps, list)
    lines = [
        "# Theorem S-expression Trace",
        "",
        f"- theorem: `{trace['theorem']}`",
        f"- split: `{trace['split']}`",
        f"- source file: `{trace['file_path']}`",
        f"- dataset rows: `{trace['dataset_row_count']}`",
        f"- validated S-expression rows: `{trace['cached_row_count']}`",
        f"- complete: `{'yes' if trace['complete'] else 'no'}`",
    ]
    missing = trace["missing_row_indices"]
    if missing:
        assert isinstance(missing, list)
        lines.append("- missing row indices: " + ", ".join(f"`{value}`" for value in missing))

    for step in steps:
        assert isinstance(step, dict)
        lines.extend(
            [
                "",
                f"## Step {step['step']} — dataset row {step['row_index']}",
                "",
                f"- tactic: `{step['tactic']}`",
                f"- alignment: `{step['alignment_kind']}`",
                f"- pending goals before tactic: `{step['pending_goal_count']}`",
                "- Pantograph target matches dataset target: "
                f"`{'yes' if step['target_state_matches_invocation'] else 'no'}`",
                "- hypothesis names match: "
                f"`{'yes' if step['hypothesis_names_match'] else 'no'}`",
            ]
        )
        transition = step["transition_matches_next_dataset_row"]
        if transition is not None:
            lines.append(
                "- target equals next recorded theorem state: "
                f"`{'yes' if transition else 'no'}`"
            )

        lines.extend(
            [
                "",
                "### Proof state before tactic",
                "",
                _fence(str(step["dataset_state"]), "lean"),
                "",
                "### Goal S-expression",
                "",
                _fence(str(step["goal_sexp"]), "lisp"),
                "",
                "### Hypothesis S-expressions",
                "",
            ]
        )
        hypotheses = step["hyp_sexps"]
        assert isinstance(hypotheses, list)
        if hypotheses:
            for hypothesis in hypotheses:
                assert isinstance(hypothesis, dict)
                lines.extend(
                    [
                        f"#### `{hypothesis.get('name', '_')}`",
                        "",
                        _fence(str(hypothesis.get("sexp", "")), "lisp"),
                        "",
                    ]
                )
        else:
            lines.extend(["_No local hypotheses._", ""])

        lines.extend(
            [
                "### Tactic",
                "",
                _fence(str(step["tactic"]), "lean"),
                "",
                "### Proof state after tactic",
                "",
                _fence(str(step["dataset_target_state"]), "lean"),
            ]
        )

    return "\n".join(lines) + "\n"


def write_theorem_trace(
    trace: dict[str, object],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_theorem_filename(str(trace['theorem']))}_{trace['split']}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(theorem_trace_markdown(trace), encoding="utf-8")
    return json_path, markdown_path
