from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .dataset import DATASET_NAME, DatasetRow, canonicalize_split_name, iter_dataset_rows
from .graph import goal_state_to_proof_state, patch_pantograph_for_sexp
from .preparation import SExprCache, _fix_pp_for_goal_start
from .reporting import console_print
from .state import parse_state


EXTRACTION_VERSION = SExprCache.EXTRACTOR_VERSION


@dataclass(frozen=True)
class SExprExtractionConfig:
    prepared_root: Path
    dataset_name: str = DATASET_NAME
    splits: tuple[str, ...] = ("train", "val", "test")
    project_path: str = "maths_ai/lean_mathlib"
    imports: tuple[str, ...] = ("Init", "Mathlib")
    sample_per_split: int | None = None
    resume: bool = True
    require_solved_theorem: bool = True


class TheoremReplayError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        phase: str,
        theorem: str,
        step_index: int | None = None,
        row_index: int | None = None,
    ) -> None:
        self.phase = phase
        self.theorem = theorem
        self.step_index = step_index
        self.row_index = row_index
        super().__init__(message)


def _expanded_hypothesis_names(state: str) -> list[str]:
    """Return local names, expanding Lean's grouped ``a b : T`` display."""
    parsed = parse_state(state)
    names: list[str] = []
    for hypothesis in parsed.hypotheses:
        grouped = hypothesis.name.split()
        names.extend(grouped or [hypothesis.name])
    return names


def _group_rows(rows: Iterable[DatasetRow]) -> dict[str, list[DatasetRow]]:
    grouped: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        grouped[row.theorem.strip()].append(row)

    result: dict[str, list[DatasetRow]] = {}
    for theorem, theorem_rows in grouped.items():
        ordered = sorted(theorem_rows, key=lambda row: row.row_index)
        indices = [row.row_index for row in ordered]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Theorem '{theorem}' contains duplicate dataset row indices.")
        result[theorem] = ordered
    return result


def _theorem_is_cached(
    cache: SExprCache,
    rows: list[DatasetRow],
) -> bool:
    return all(
        (
            record := cache.load_for_row(
                row,
                step_index=step_index,
                extractor_version=EXTRACTION_VERSION,
            )
        )
        is not None
        and isinstance(record.get("goal_sexp"), str)
        and isinstance(record.get("hyp_sexps"), list)
        for step_index, row in enumerate(rows)
    )


def _make_record(
    *,
    row: DatasetRow,
    step_index: int,
    goal_state,
) -> dict[str, object]:
    if not goal_state.goals:
        raise TheoremReplayError(
            "The replayed proof was solved before the dataset row was reached.",
            phase="state_alignment",
            theorem=row.theorem,
            step_index=step_index,
            row_index=row.row_index,
        )

    expected_goal_count = sum(
        1
        for line in row.state.splitlines()
        if line.lstrip().startswith(("⊢", "|-"))
    )
    if expected_goal_count != 1:
        raise TheoremReplayError(
            f"Dataset row has {expected_goal_count} textual goals; exactly one is required.",
            phase="state_alignment",
            theorem=row.theorem,
            step_index=step_index,
            row_index=row.row_index,
        )

    expected_names = _expanded_hypothesis_names(row.state)
    active_goal = goal_state.goals[0]
    actual_names = [variable.name or "_" for variable in active_goal.variables]
    if len(actual_names) != len(expected_names):
        raise TheoremReplayError(
            "Hypothesis-count mismatch at replay step "
            f"{step_index}: dataset={len(expected_names)}, pantograph={len(actual_names)}.",
            phase="state_alignment",
            theorem=row.theorem,
            step_index=step_index,
            row_index=row.row_index,
        )

    text_state, hyp_sexps, goal_sexp = goal_state_to_proof_state(goal_state)
    if not isinstance(goal_sexp, str) or not goal_sexp:
        raise TheoremReplayError(
            "Pantograph returned no S-expression for the active goal.",
            phase="sexpr_capture",
            theorem=row.theorem,
            step_index=step_index,
            row_index=row.row_index,
        )
    if len(hyp_sexps) != len(expected_names):
        raise TheoremReplayError(
            "Captured hypothesis S-expression count does not match the dataset state.",
            phase="sexpr_capture",
            theorem=row.theorem,
            step_index=step_index,
            row_index=row.row_index,
        )

    return {
        "schema_version": SExprCache.SCHEMA_VERSION,
        "extractor_version": EXTRACTION_VERSION,
        "dataset": row.dataset_name,
        "split": row.split,
        "row_index": row.row_index,
        "theorem": row.theorem,
        "step_index": step_index,
        "state_sha256": SExprCache.row_state_sha256(row),
        "tactic_sha256": SExprCache.row_tactic_sha256(row),
        "tactic": row.tactic,
        "pending_goal_count": len(goal_state.goals),
        "hypothesis_count": len(hyp_sexps),
        "hypothesis_names_match": actual_names == expected_names,
        "goal_sexp": goal_sexp,
        "hyp_sexps": [
            {"name": name, "sexp": sexp} for name, sexp in hyp_sexps
        ],
        "text_state": text_state,
    }


async def replay_theorem_rows(
    server,
    theorem: str,
    rows: list[DatasetRow],
    *,
    require_solved: bool = True,
) -> list[dict[str, object]]:
    """Replay one theorem once and capture the state immediately before each tactic."""
    if not theorem:
        raise TheoremReplayError(
            "Dataset theorem name is empty.",
            phase="theorem_identity",
            theorem=theorem,
        )
    if not rows:
        return []
    if any(row.theorem.strip() != theorem for row in rows):
        raise TheoremReplayError(
            "Rows from different theorems were passed to one replay.",
            phase="theorem_identity",
            theorem=theorem,
        )

    try:
        inspection = await server.env_inspect_async(theorem)
        theorem_type = inspection["type"]["pp"]
    except Exception as exc:
        raise TheoremReplayError(
            f"Could not inspect theorem '{theorem}': {exc}",
            phase="theorem_inspect",
            theorem=theorem,
        ) from exc

    try:
        goal_state = await server.goal_start_async(_fix_pp_for_goal_start(theorem_type))
    except Exception as exc:
        raise TheoremReplayError(
            f"Could not start theorem '{theorem}': {exc}",
            phase="theorem_start",
            theorem=theorem,
        ) from exc

    # A traced tactic state begins inside the theorem declaration, after its
    # declaration binders have become local hypotheses. Recreate precisely the
    # number and order shown in the first dataset state.
    first_names = _expanded_hypothesis_names(rows[0].state)
    for name in first_names:
        tactic = f"intro {name}" if name and not any(c.isspace() for c in name) else "intro"
        try:
            goal_state = await server.goal_tactic_async(goal_state, tactic)
        except Exception:
            try:
                goal_state = await server.goal_tactic_async(goal_state, "intro")
            except Exception as exc:
                raise TheoremReplayError(
                    "Could not recreate the theorem's initial local context: "
                    f"failed while introducing '{name}'.",
                    phase="initial_context",
                    theorem=theorem,
                    step_index=0,
                    row_index=rows[0].row_index,
                ) from exc

    records: list[dict[str, object]] = []
    for step_index, row in enumerate(rows):
        records.append(
            _make_record(row=row, step_index=step_index, goal_state=goal_state)
        )
        try:
            goal_state = await server.goal_tactic_async(goal_state, row.tactic)
        except Exception as exc:
            raise TheoremReplayError(
                f"Tactic replay failed at step {step_index}: {row.tactic!r}: {exc}",
                phase="tactic_replay",
                theorem=theorem,
                step_index=step_index,
                row_index=row.row_index,
            ) from exc

    if require_solved and not goal_state.is_solved:
        raise TheoremReplayError(
            f"Replay ended with {len(goal_state.goals)} unsolved goal(s).",
            phase="incomplete_theorem",
            theorem=theorem,
            step_index=len(rows),
        )
    return records


def _failure_record(exc: Exception, theorem: str, rows: list[DatasetRow]) -> dict[str, object]:
    return {
        "theorem": theorem,
        "split": rows[0].split if rows else None,
        "row_indices": [row.row_index for row in rows],
        "failed_row_count": len(rows),
        "phase": getattr(exc, "phase", "unexpected_error"),
        "step_index": getattr(exc, "step_index", None),
        "row_index": getattr(exc, "row_index", None),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# S-expression Extraction Summary",
        "",
        f"- extractor: `{summary['extractor_version']}`",
        f"- dataset: `{summary['dataset']}`",
        f"- attempted rows: `{summary['attempted_rows']}`",
        f"- covered rows: `{summary['covered_rows']}`",
        f"- failed rows: `{summary['failed_rows']}`",
        f"- coverage: `{float(summary['coverage']):.4%}`",
        "",
        "| Split | Theorems | Cached | Extracted | Failed | Rows | Covered | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    manifests = summary["manifests"]
    for split in summary["splits"]:
        manifest = manifests[split]
        lines.append(
            f"| {split} | {manifest['attempted_theorems']} | "
            f"{manifest['cached_theorems']} | {manifest['extracted_theorems']} | "
            f"{manifest['failed_theorems']} | {manifest['attempted_rows']} | "
            f"{manifest['covered_rows']} | {float(manifest['coverage']):.4%} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def extract_split_with_server(
    server,
    *,
    rows: list[DatasetRow],
    cache: SExprCache,
    prepared_root: Path,
    split: str,
    resume: bool = True,
    require_solved: bool = True,
) -> dict[str, object]:
    theorem_groups = _group_rows(rows)
    report_root = Path(prepared_root) / "sexpr_extraction"
    failure_path = report_root / "failures" / f"{split}.jsonl"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text("", encoding="utf-8")

    cached_theorems = 0
    extracted_theorems = 0
    failed_theorems = 0
    cached_rows = 0
    extracted_rows = 0
    failed_rows = 0
    failure_phases: Counter[str] = Counter()

    console_print(
        f"  [{split}] theorem replay: {len(theorem_groups)} theorems, {len(rows)} rows"
    )
    for theorem_index, (theorem, theorem_rows) in enumerate(theorem_groups.items(), 1):
        if resume and _theorem_is_cached(cache, theorem_rows):
            cached_theorems += 1
            cached_rows += len(theorem_rows)
            continue

        try:
            records = await replay_theorem_rows(
                server,
                theorem,
                theorem_rows,
                require_solved=require_solved,
            )
            if len(records) != len(theorem_rows):
                raise TheoremReplayError(
                    "Replay produced a different number of states than dataset rows.",
                    phase="state_count",
                    theorem=theorem,
                )
            # Commit only after the entire theorem replay and all alignment
            # checks have succeeded. A failed theorem never leaves a partial
            # set of apparently valid cache records.
            for row, record in zip(theorem_rows, records):
                cache.save(row.split, row.row_index, record)
            extracted_theorems += 1
            extracted_rows += len(theorem_rows)
        except Exception as exc:
            failure = _failure_record(exc, theorem, theorem_rows)
            with failure_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            failed_theorems += 1
            failed_rows += len(theorem_rows)
            failure_phases[str(failure["phase"])] += 1
            console_print(
                f"  [{split}] replay failed for {theorem}: "
                f"{failure['phase']}: {failure['error']}"
            )

        if theorem_index == 1 or theorem_index % 100 == 0 or theorem_index == len(theorem_groups):
            console_print(
                f"  [{split}] {theorem_index}/{len(theorem_groups)} theorems | "
                f"cached={cached_theorems} extracted={extracted_theorems} "
                f"failed={failed_theorems}"
            )

    attempted_rows = len(rows)
    covered_rows = cached_rows + extracted_rows
    manifest: dict[str, object] = {
        "schema_version": SExprCache.SCHEMA_VERSION,
        "extractor_version": EXTRACTION_VERSION,
        "dataset": rows[0].dataset_name if rows else None,
        "split": split,
        "attempted_theorems": len(theorem_groups),
        "cached_theorems": cached_theorems,
        "extracted_theorems": extracted_theorems,
        "failed_theorems": failed_theorems,
        "attempted_rows": attempted_rows,
        "cached_rows": cached_rows,
        "extracted_rows": extracted_rows,
        "failed_rows": failed_rows,
        "covered_rows": covered_rows,
        "coverage": covered_rows / attempted_rows if attempted_rows else 0.0,
        "failure_phases": dict(sorted(failure_phases.items())),
        "failure_log": str(failure_path),
    }
    _write_json(report_root / "manifests" / f"{split}.json", manifest)
    return manifest


async def extract_sexpressions(
    config: SExprExtractionConfig,
    *,
    server_factory=None,
) -> dict[str, object]:
    if server_factory is None:
        patch_pantograph_for_sexp()
        from pantograph.server import Server

        factory = Server.create
    else:
        factory = server_factory
    server = await factory(
        project_path=config.project_path,
        imports=list(config.imports),
        options={"printExprAST": True},
    )
    cache = SExprCache(config.prepared_root, config.project_path, enabled=True)
    manifests: dict[str, dict[str, object]] = {}
    try:
        for raw_split in config.splits:
            split = canonicalize_split_name(raw_split)
            rows = list(
                iter_dataset_rows(
                    dataset_name=config.dataset_name,
                    split=split,
                    sample_limit=config.sample_per_split,
                )
            )
            manifests[split] = await extract_split_with_server(
                server,
                rows=rows,
                cache=cache,
                prepared_root=config.prepared_root,
                split=split,
                resume=config.resume,
                require_solved=config.require_solved_theorem,
            )
    finally:
        try:
            await server.shutdown_async()
        except Exception:
            close = getattr(server, "_close", None)
            if close is not None:
                close()

    attempted_rows = sum(int(item["attempted_rows"]) for item in manifests.values())
    covered_rows = sum(int(item["covered_rows"]) for item in manifests.values())
    summary: dict[str, object] = {
        "schema_version": SExprCache.SCHEMA_VERSION,
        "extractor_version": EXTRACTION_VERSION,
        "dataset": config.dataset_name,
        "prepared_root": str(config.prepared_root),
        "splits": list(manifests),
        "manifests": manifests,
        "attempted_rows": attempted_rows,
        "covered_rows": covered_rows,
        "failed_rows": attempted_rows - covered_rows,
        "coverage": covered_rows / attempted_rows if attempted_rows else 0.0,
    }
    report_root = Path(config.prepared_root) / "sexpr_extraction"
    _write_json(report_root / "summary.json", summary)
    _write_summary_markdown(report_root / "summary.md", summary)
    return summary
