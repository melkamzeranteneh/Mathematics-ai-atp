from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .dataset import DATASET_NAME, DatasetRow, canonicalize_split_name, iter_dataset_rows
from .preparation import SExprCache
from .reporting import console_print
from .state import parse_state


EXTRACTION_VERSION = SExprCache.EXTRACTOR_VERSION
DATASET_MATHLIB_COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"
PANTOGRAPH_COMMIT = "22ddfaaf2124d323dec59220f567273f01623458"


@dataclass(frozen=True)
class SExprExtractionConfig:
    prepared_root: Path
    source_root: Path
    pantograph_repl: Path
    dataset_name: str = DATASET_NAME
    splits: tuple[str, ...] = ("train", "val", "test")
    sample_per_split: int | None = None
    resume: bool = True
    expected_commit: str = DATASET_MATHLIB_COMMIT
    expected_pantograph_commit: str = PANTOGRAPH_COMMIT
    server_startup_timeout: int = 120
    file_timeout: int = 600
    buffer_limit: int = 256 * 1024 * 1024
    verify_source_commit: bool = True


class SourceExtractionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        phase: str,
        theorem: str = "",
        row_index: int | None = None,
    ) -> None:
        self.phase = phase
        self.theorem = theorem
        self.row_index = row_index
        super().__init__(message)


class PantographInvocationClient:
    """Small client for the Lean-4.10-compatible Pantograph JSON REPL.

    We intentionally do not import the Python Pantograph package: the project's
    regular environment may use another Lean/Pantograph version.  ``lake env``
    selects the exact toolchain recorded by the dataset checkout.
    """

    def __init__(
        self,
        *,
        source_root: Path,
        pantograph_repl: Path,
        startup_timeout: int = 120,
        file_timeout: int = 600,
        buffer_limit: int = 256 * 1024 * 1024,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.pantograph_repl = Path(pantograph_repl).resolve()
        self.startup_timeout = startup_timeout
        self.file_timeout = file_timeout
        self.buffer_limit = buffer_limit
        self.proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self.stderr_tail: list[str] = []

    async def start(self) -> "PantographInvocationClient":
        if self.proc is not None:
            return self
        self.proc = await asyncio.create_subprocess_exec(
            "lake",
            "env",
            "stdbuf",
            "-oL",
            str(self.pantograph_repl),
            "Init",
            cwd=self.source_root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=self.buffer_limit,
        )
        assert self.proc.stdout is not None
        try:
            ready = await asyncio.wait_for(
                self.proc.stdout.readline(), timeout=self.startup_timeout
            )
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RuntimeError("Pantograph did not become ready in time.") from exc
        if ready.decode("utf-8", errors="replace").strip() != "ready.":
            await self.close()
            raise RuntimeError(f"Pantograph emitted an invalid ready signal: {ready!r}")
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self.call("options.set", {"printExprAST": True}, timeout=self.startup_timeout)
        return self

    async def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        while line := await self.proc.stderr.readline():
            self.stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())
            del self.stderr_tail[:-30]

    async def call(
        self, command: str, payload: dict[str, object], *, timeout: int
    ) -> dict[str, object]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Pantograph client is not running.")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.proc.stdin.write(f"{command} {encoded}\n".encode("utf-8"))
        await self.proc.stdin.drain()
        try:
            raw = await asyncio.wait_for(self.proc.stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RuntimeError(f"Pantograph command '{command}' timed out.") from exc
        if not raw:
            detail = "\n".join(self.stderr_tail[-10:])
            await self.close()
            raise RuntimeError(f"Pantograph exited during '{command}'. {detail}")
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            await self.close()
            raise RuntimeError(f"Pantograph returned invalid JSON for '{command}'.") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Pantograph returned a non-object for '{command}'.")
        if "error" in result:
            detail = json.dumps(result, ensure_ascii=False, sort_keys=True)
            raise RuntimeError(f"Pantograph '{command}' failed: {detail}")
        return result

    async def process_file(self, file_path: str) -> list[dict[str, object]]:
        if self.proc is None:
            await self.start()
        absolute = (self.source_root / file_path).resolve()
        try:
            absolute.relative_to(self.source_root)
        except ValueError as exc:
            raise RuntimeError(f"Dataset file escapes the source checkout: {file_path}") from exc
        descriptor, output_name = tempfile.mkstemp(
            prefix="maths_ai_invocations_", suffix=".json"
        )
        os.close(descriptor)
        output_path = Path(output_name)
        try:
            await self.call(
                "frontend.process",
                {
                    "fileName": str(absolute),
                    "invocations": True,
                    "sorrys": False,
                    "outputFile": str(output_path),
                },
                timeout=self.file_timeout,
            )
            try:
                with output_path.open(encoding="utf-8") as handle:
                    result = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "Pantograph did not write a valid frontend trace."
                ) from exc
            if not isinstance(result, dict):
                raise RuntimeError("Pantograph frontend trace is not an object.")
            units = result.get("units")
            if not isinstance(units, list):
                raise RuntimeError("Pantograph frontend trace has no compilation units.")
            return units
        finally:
            output_path.unlink(missing_ok=True)

    async def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
            try:
                await proc.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
        if proc is not None and proc.returncode is None:
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=2)
            except asyncio.TimeoutError:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=5)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    await proc.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
            self._stderr_task = None


def _normalized_text(text: str) -> str:
    normalized = "\n".join(
        line.rstrip()
        for line in text.replace("\r\n", "\n").strip().splitlines()
    )
    if normalized.lower() in {"no goals", "no goals to be solved"}:
        return ""
    return normalized


def _normalized_tactic(text: str) -> str:
    # LeanDojo adds HTML-like links around referenced declarations, and the
    # link text may be fully qualified even when Lean's tactic pretty-printer
    # chooses the short name. These decorations are not part of Lean syntax.
    text = re.sub(r"</?a(?:\s[^>]*)?>", "", text)
    text = re.sub(
        r"(?<![\w'])(?:[\w'][\w'!?]*\.)+([\w'][\w'!?]*)",
        r"\1",
        text,
    )
    return " ".join(text.split())


def _expanded_hypothesis_names(state: str) -> list[str]:
    names: list[str] = []
    for hypothesis in parse_state(state).hypotheses:
        names.extend(hypothesis.name.split() or [hypothesis.name])
    return names


def _group_rows_by_file(rows: Iterable[DatasetRow]) -> dict[str, dict[str, list[DatasetRow]]]:
    grouped: dict[str, dict[str, list[DatasetRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row.file_path:
            raise SourceExtractionError(
                "Dataset row has no file_path metadata.",
                phase="source_metadata",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        if not row.repo_commit:
            raise SourceExtractionError(
                "Dataset row has no commit metadata.",
                phase="source_metadata",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        grouped[row.file_path][row.theorem].append(row)
    return {
        file_path: {
            theorem: sorted(theorem_rows, key=lambda item: item.row_index)
            for theorem, theorem_rows in theorem_groups.items()
        }
        for file_path, theorem_groups in grouped.items()
    }


def _unit_source(source_bytes: bytes, unit: dict[str, object]) -> str:
    boundary = unit.get("boundary")
    if not isinstance(boundary, list) or len(boundary) != 2:
        return ""
    try:
        return source_bytes[int(boundary[0]) : int(boundary[1])].decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""


def _unit_declares_theorem(source: str, theorem: str) -> bool:
    if not theorem:
        return False
    short_name = theorem.rsplit(".", 1)[-1].strip("«»")
    declaration = re.compile(
        rf"\b(?:theorem|lemma)\s+(?:[A-Za-z0-9_'.]+\.)?«?{re.escape(short_name)}»?(?=\s|[:(])"
    )
    return declaration.search(source) is not None


def _candidate_invocations(
    *, source_bytes: bytes, units: list[dict[str, object]], theorem: str
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    ordinal = 0
    for unit_index, unit in enumerate(units):
        source = _unit_source(source_bytes, unit)
        invocations = unit.get("invocations") or []
        if not isinstance(invocations, list):
            continue
        is_theorem_unit = _unit_declares_theorem(source, theorem)
        for invocation_index, invocation in enumerate(invocations):
            if is_theorem_unit and isinstance(invocation, dict):
                candidates.append(
                    {
                        "ordinal": ordinal,
                        "unit_index": unit_index,
                        "invocation_index": invocation_index,
                        "unit_boundary": unit.get("boundary"),
                        "invocation": invocation,
                    }
                )
            ordinal += 1
    return candidates


def _invocation_matches(row: DatasetRow, candidate: dict[str, object]) -> bool:
    invocation = candidate["invocation"]
    assert isinstance(invocation, dict)
    if _normalized_text(str(invocation.get("goalBefore", ""))) != _normalized_text(row.state):
        return False
    if _normalized_tactic(str(invocation.get("tactic", ""))) != _normalized_tactic(row.tactic):
        return False
    if row.target_state and _normalized_text(str(invocation.get("goalAfter", ""))) != _normalized_text(row.target_state):
        return False
    return True


def _align_rows(
    rows: list[DatasetRow], candidates: list[dict[str, object]]
) -> list[dict[str, object]]:
    choices = [[item for item in candidates if _invocation_matches(row, item)] for row in rows]
    for row, matches in zip(rows, choices):
        if not matches:
            raise SourceExtractionError(
                "No original tactic invocation matches state + tactic + target_state.",
                phase="invocation_alignment",
                theorem=row.theorem,
                row_index=row.row_index,
            )

    solutions: list[list[dict[str, object]]] = []

    def search(index: int, previous: int, selected: list[dict[str, object]]) -> None:
        if len(solutions) >= 2:
            return
        if index == len(rows):
            solutions.append(selected.copy())
            return
        for candidate in choices[index]:
            ordinal = int(candidate["ordinal"])
            if ordinal > previous:
                selected.append(candidate)
                search(index + 1, ordinal, selected)
                selected.pop()

    search(0, -1, [])
    if not solutions:
        raise SourceExtractionError(
            "Matching invocations exist but cannot be aligned in dataset order.",
            phase="invocation_order",
            theorem=rows[0].theorem,
        )
    if len(solutions) > 1:
        raise SourceExtractionError(
            "More than one source invocation sequence matches these rows; refusing a guessed cache.",
            phase="ambiguous_invocation",
            theorem=rows[0].theorem,
        )
    return solutions[0]


def _make_record(row: DatasetRow, candidate: dict[str, object]) -> dict[str, object]:
    invocation = candidate["invocation"]
    assert isinstance(invocation, dict)
    capture_error = invocation.get("captureError")
    if capture_error:
        raise SourceExtractionError(
            f"Pantograph could not serialize this tactic's proof state: {capture_error}",
            phase="sexpr_capture",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    goals = invocation.get("goalsBefore")
    if not isinstance(goals, list) or len(goals) != 1:
        raise SourceExtractionError(
            f"Invocation has {len(goals) if isinstance(goals, list) else 'invalid'} active goals; expected one.",
            phase="goal_cardinality",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    goal = goals[0]
    if not isinstance(goal, dict):
        raise SourceExtractionError("Serialized goal is invalid.", phase="sexpr_capture", theorem=row.theorem)
    target = goal.get("target")
    variables = goal.get("vars")
    if not isinstance(target, dict) or not isinstance(target.get("sexp"), str):
        raise SourceExtractionError("Goal S-expression is missing.", phase="sexpr_capture", theorem=row.theorem)
    if not isinstance(variables, list):
        raise SourceExtractionError("Goal variables are missing.", phase="sexpr_capture", theorem=row.theorem)

    hyp_sexps: list[dict[str, str]] = []
    actual_names: list[str] = []
    for variable in variables:
        if not isinstance(variable, dict) or not isinstance(variable.get("type"), dict):
            raise SourceExtractionError("Variable serialization is invalid.", phase="sexpr_capture", theorem=row.theorem)
        sexp = variable["type"].get("sexp")
        if not isinstance(sexp, str):
            raise SourceExtractionError("Variable type S-expression is missing.", phase="sexpr_capture", theorem=row.theorem)
        name = str(variable.get("userName", "_"))
        actual_names.append(name)
        hyp_sexps.append({"name": name, "sexp": sexp})

    expected_names = _expanded_hypothesis_names(row.state)
    return {
        "schema_version": SExprCache.SCHEMA_VERSION,
        "extractor_version": EXTRACTION_VERSION,
        "dataset": row.dataset_name,
        "split": row.split,
        "row_index": row.row_index,
        "theorem": row.theorem,
        "repo_url": row.repo_url,
        "repo_commit": row.repo_commit,
        "pantograph_commit": PANTOGRAPH_COMMIT,
        "file_path": row.file_path,
        "unit_index": candidate["unit_index"],
        "unit_boundary": candidate["unit_boundary"],
        "invocation_index": candidate["invocation_index"],
        "state_sha256": SExprCache.row_state_sha256(row),
        "tactic_sha256": SExprCache.row_tactic_sha256(row),
        "target_state_sha256": SExprCache.row_target_state_sha256(row),
        "tactic": row.tactic,
        "pending_goal_count": len(goals),
        "hypothesis_count": len(hyp_sexps),
        "hypothesis_names_match": actual_names == expected_names,
        "goal_sexp": target["sexp"],
        "hyp_sexps": hyp_sexps,
        "text_state": str(invocation.get("goalBefore", "")),
        "text_target_state": str(invocation.get("goalAfter", "")),
    }


def _row_is_cached(cache: SExprCache, row: DatasetRow) -> bool:
    record = cache.load_for_row(row, extractor_version=EXTRACTION_VERSION)
    return (
        record is not None
        and isinstance(record.get("goal_sexp"), str)
        and isinstance(record.get("hyp_sexps"), list)
    )


def _failure_record(exc: Exception, file_path: str, theorem: str, rows: list[DatasetRow]) -> dict[str, object]:
    return {
        "file_path": file_path,
        "theorem": theorem,
        "split": rows[0].split if rows else None,
        "row_indices": [row.row_index for row in rows],
        "failed_row_count": len(rows),
        "phase": getattr(exc, "phase", "unexpected_error"),
        "row_index": getattr(exc, "row_index", None),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# S-expression Extraction Summary",
        "",
        f"- extractor: `{summary['extractor_version']}`",
        f"- dataset: `{summary['dataset']}`",
        f"- source commit: `{summary['source_commit']}`",
        f"- attempted rows: `{summary['attempted_rows']}`",
        f"- covered rows: `{summary['covered_rows']}`",
        f"- failed rows: `{summary['failed_rows']}`",
        f"- coverage: `{float(summary['coverage']):.4%}`",
        "",
        "| Split | Files | Theorems | Cached rows | Extracted rows | Failed rows | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    manifests = summary["manifests"]
    for split in summary["splits"]:
        manifest = manifests[split]
        lines.append(
            f"| {split} | {manifest['attempted_files']} | {manifest['attempted_theorems']} | "
            f"{manifest['cached_rows']} | {manifest['extracted_rows']} | "
            f"{manifest['failed_rows']} | {float(manifest['coverage']):.4%} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def extract_split_with_client(
    client,
    *,
    rows: list[DatasetRow],
    cache: SExprCache,
    prepared_root: Path,
    source_root: Path,
    split: str,
    expected_commit: str = DATASET_MATHLIB_COMMIT,
    resume: bool = True,
) -> dict[str, object]:
    report_root = Path(prepared_root) / "sexpr_extraction"
    failure_path = report_root / "failures" / f"{split}.jsonl"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text("", encoding="utf-8")

    try:
        file_groups = _group_rows_by_file(rows)
    except Exception as exc:
        file_groups = {}
        failure = _failure_record(exc, "", getattr(exc, "theorem", ""), rows)
        failure_path.write_text(json.dumps(failure, ensure_ascii=False) + "\n", encoding="utf-8")
        failure_phases = Counter({str(failure["phase"]): len(rows)})
        cached_rows = extracted_rows = compiled_files = 0
        failed_rows = len(rows)
    else:
        cached_rows = sum(1 for row in rows if resume and _row_is_cached(cache, row))
        extracted_rows = failed_rows = compiled_files = 0
        failure_phases: Counter[str] = Counter()
        console_print(f"  [{split}] source extraction: {len(file_groups)} files, {len(rows)} rows")

        for file_index, (file_path, theorem_groups) in enumerate(file_groups.items(), 1):
            file_rows = [row for theorem_rows in theorem_groups.values() for row in theorem_rows]
            uncached_file_rows = [row for row in file_rows if not (resume and _row_is_cached(cache, row))]
            if not uncached_file_rows:
                continue
            bad_commits = sorted({row.repo_commit for row in file_rows if row.repo_commit != expected_commit})
            if bad_commits:
                exc = SourceExtractionError(
                    f"Dataset row commit does not match extractor checkout: {bad_commits}",
                    phase="commit_mismatch",
                )
                units = None
            else:
                try:
                    units = await client.process_file(file_path)
                    compiled_files += 1
                    source_bytes = (Path(source_root) / file_path).read_bytes()
                except Exception as caught:
                    exc = SourceExtractionError(str(caught), phase="file_compile")
                    units = None

            for theorem, theorem_rows in theorem_groups.items():
                uncached_rows = [row for row in theorem_rows if not (resume and _row_is_cached(cache, row))]
                if not uncached_rows:
                    continue
                try:
                    if units is None:
                        raise exc
                    candidates = _candidate_invocations(
                        source_bytes=source_bytes, units=units, theorem=theorem
                    )
                    if not candidates:
                        raise SourceExtractionError(
                            "Could not identify the theorem's compilation unit.",
                            phase="theorem_identity",
                            theorem=theorem,
                        )
                    aligned = _align_rows(theorem_rows, candidates)
                    records = [
                        _make_record(row, candidate)
                        for row, candidate in zip(theorem_rows, aligned)
                    ]
                    # Validate every row in the theorem before committing any
                    # of them, so a capture error cannot leave a partial group.
                    for row, record in zip(theorem_rows, records):
                        if row in uncached_rows:
                            cache.save(row.split, row.row_index, record)
                            extracted_rows += 1
                except Exception as caught:
                    failure = _failure_record(caught, file_path, theorem, uncached_rows)
                    with failure_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
                    failed_rows += len(uncached_rows)
                    failure_phases[str(failure["phase"])] += len(uncached_rows)
                    console_print(f"  [{split}] failed {theorem}: {failure['phase']}: {failure['error']}")

            if file_index == 1 or file_index % 25 == 0 or file_index == len(file_groups):
                console_print(
                    f"  [{split}] {file_index}/{len(file_groups)} files | cached={cached_rows} "
                    f"extracted={extracted_rows} failed={failed_rows}"
                )

    attempted_rows = len(rows)
    covered_rows = cached_rows + extracted_rows
    manifest: dict[str, object] = {
        "schema_version": SExprCache.SCHEMA_VERSION,
        "extractor_version": EXTRACTION_VERSION,
        "dataset": rows[0].dataset_name if rows else None,
        "source_commit": expected_commit,
        "split": split,
        "attempted_files": len(file_groups),
        "compiled_files": compiled_files,
        "attempted_theorems": sum(len(groups) for groups in file_groups.values()),
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


def _git_head(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


async def extract_sexpressions(config: SExprExtractionConfig, *, client_factory=None) -> dict[str, object]:
    source_root = Path(config.source_root).resolve()
    repl = Path(config.pantograph_repl).resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"Mathlib source checkout does not exist: {source_root}")
    if client_factory is None and not repl.is_file():
        raise RuntimeError(f"Patched Pantograph REPL does not exist: {repl}")
    if config.verify_source_commit:
        actual_commit = _git_head(source_root)
        if actual_commit != config.expected_commit:
            raise RuntimeError(
                f"Mathlib checkout is {actual_commit}, but dataset requires {config.expected_commit}."
            )
    if client_factory is None:
        try:
            pantograph_root = repl.parents[3]
            actual_pantograph_commit = _git_head(pantograph_root)
        except (IndexError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Could not identify the Pantograph checkout containing {repl}."
            ) from exc
        if actual_pantograph_commit != config.expected_pantograph_commit:
            raise RuntimeError(
                "Pantograph checkout is "
                f"{actual_pantograph_commit}, but extraction requires "
                f"{config.expected_pantograph_commit}."
            )

    if client_factory is None:
        client = PantographInvocationClient(
            source_root=source_root,
            pantograph_repl=repl,
            startup_timeout=config.server_startup_timeout,
            file_timeout=config.file_timeout,
            buffer_limit=config.buffer_limit,
        )
    else:
        client = client_factory(config)
        if asyncio.iscoroutine(client):
            client = await client
    cache = SExprCache(config.prepared_root, str(source_root), enabled=True)
    manifests: dict[str, dict[str, object]] = {}
    started = False
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
            if not started:
                await client.start()
                started = True
            manifests[split] = await extract_split_with_client(
                client,
                rows=rows,
                cache=cache,
                prepared_root=config.prepared_root,
                source_root=source_root,
                split=split,
                expected_commit=config.expected_commit,
                resume=config.resume,
            )
    finally:
        if started:
            await client.close()

    attempted_rows = sum(int(item["attempted_rows"]) for item in manifests.values())
    covered_rows = sum(int(item["covered_rows"]) for item in manifests.values())
    summary: dict[str, object] = {
        "schema_version": SExprCache.SCHEMA_VERSION,
        "extractor_version": EXTRACTION_VERSION,
        "dataset": config.dataset_name,
        "source_root": str(source_root),
        "source_commit": config.expected_commit,
        "pantograph_commit": config.expected_pantograph_commit,
        "pantograph_repl": str(repl),
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
