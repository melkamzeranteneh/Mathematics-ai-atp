from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..dataset import canonicalize_split_name
from ..preparation import ActionTraceCache, SExprCache
from ..reporting import console_print
from .compiler import ActionTargetError, compile_action_trace


@dataclass(frozen=True)
class ActionTargetAuditConfig:
    cache_root: Path
    output_dir: Path
    splits: tuple[str, ...] = ("train",)
    max_items_per_split: int | None = None
    sample_limit_per_category: int = 20
    force: bool = False

    def normalized(self) -> "ActionTargetAuditConfig":
        splits = tuple(canonicalize_split_name(split) for split in self.splits)
        if not splits:
            raise ValueError("At least one split is required.")
        if self.max_items_per_split is not None and self.max_items_per_split < 1:
            raise ValueError("max_items_per_split must be positive when provided.")
        if self.sample_limit_per_category < 0:
            raise ValueError("sample_limit_per_category cannot be negative.")
        return ActionTargetAuditConfig(
            cache_root=self.cache_root.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            splits=splits,
            max_items_per_split=self.max_items_per_split,
            sample_limit_per_category=self.sample_limit_per_category,
            force=self.force,
        )


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _tactic_name(tactic: str) -> str:
    parts = tactic.strip().split(maxsplit=1)
    return parts[0] if parts else "<empty>"


def _manifest_counts(cache_root: Path, split: str) -> dict[str, int | float] | None:
    path = cache_root / "action_trace_extraction_v2" / "manifests" / f"{split}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "attempted_rows",
        "cached_rows",
        "extracted_rows",
        "failed_rows",
        "covered_rows",
        "coverage",
    )
    return {key: payload[key] for key in keys if key in payload}


def _render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Structured Action Target Audit",
        "",
        f"- Cache root: `{summary['cache_root']}`",
        f"- Trace version: `{ActionTraceCache.EXTRACTOR_VERSION}`",
        "",
    ]
    for split, payload in dict(summary["splits"]).items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Trace files: `{payload['trace_file_count']}`",
                f"- Valid digest-bound traces: `{payload['valid_trace_count']}`",
                f"- Compiled targets: `{payload['compiled_count']}`",
                f"- Compilation failures: `{payload['failure_count']}`",
                f"- Rows with traced payload: `{payload['payload_row_count']}`",
                f"- Empty payload rows: `{payload['empty_payload_row_count']}`",
                f"- Compiler success rate: `{payload['compiler_success_rate']:.4%}`",
                "",
                "### Sequence lengths",
                "",
                f"- median: `{payload['sequence_lengths']['median']}`",
                f"- p95: `{payload['sequence_lengths']['p95']}`",
                f"- p99: `{payload['sequence_lengths']['p99']}`",
                f"- maximum: `{payload['sequence_lengths']['max']}`",
                "",
                "### Operations",
                "",
                "| Operation | Count |",
                "| --- | ---: |",
            ]
        )
        for operation, count in dict(payload["operation_counts"]).items():
            lines.append(f"| {operation} | {count} |")
        lines.extend(
            ["", "### Failures", "", "| Category | Count |", "| --- | ---: |"]
        )
        for category, count in dict(payload["failure_categories"]).items():
            lines.append(f"| {category} | {count} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_action_target_audit(config: ActionTargetAuditConfig) -> dict[str, object]:
    config = config.normalized()
    if not config.cache_root.is_dir():
        raise FileNotFoundError(
            f"S-expression cache root does not exist: {config.cache_root}"
        )
    output_is_populated = config.output_dir.exists() and any(
        config.output_dir.iterdir()
    )
    if output_is_populated and not config.force:
        raise FileExistsError(
            f"Output directory is not empty: {config.output_dir}. "
            "Use --force to overwrite reports."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_cache = SExprCache(config.cache_root, project_path="", enabled=True)
    action_cache = ActionTraceCache(config.cache_root, enabled=True)
    targets_path = config.output_dir / "targets.jsonl"
    samples_path = config.output_dir / "samples.jsonl"
    summary: dict[str, object] = {
        "schema_version": 1,
        "cache_root": str(config.cache_root),
        "trace_extractor_version": ActionTraceCache.EXTRACTOR_VERSION,
        "splits": {},
    }
    sample_counts: Counter[tuple[str, str]] = Counter()

    with targets_path.open("w", encoding="utf-8") as targets_handle, samples_path.open(
        "w", encoding="utf-8"
    ) as samples_handle:
        for split in config.splits:
            trace_dir = config.cache_root / split / "action_trace_v2"
            trace_files = sorted(trace_dir.glob("*.json")) if trace_dir.exists() else []
            if config.max_items_per_split is not None:
                trace_files = trace_files[: config.max_items_per_split]
            if not trace_files:
                raise FileNotFoundError(f"No action_trace_v2 sidecars found under {trace_dir}")

            valid_count = compiled_count = payload_count = empty_count = 0
            failure_categories: Counter[str] = Counter()
            operation_counts: Counter[str] = Counter()
            sequence_lengths: list[int] = []
            per_tactic: defaultdict[str, Counter[str]] = defaultdict(Counter)

            for processed, path in enumerate(trace_files, 1):
                try:
                    row_index = int(path.stem)
                except ValueError:
                    failure_categories["invalid_sidecar_filename"] += 1
                    continue
                raw_record = raw_cache.load(split, row_index)
                trace = (
                    action_cache.load_for_raw_record(split, row_index, raw_record)
                    if raw_record is not None
                    else None
                )
                if trace is None:
                    category = "invalid_or_stale_trace"
                    failure_categories[category] += 1
                    if sample_counts[(split, category)] < config.sample_limit_per_category:
                        samples_handle.write(
                            json.dumps(
                                {"split": split, "row_index": row_index, "category": category},
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        sample_counts[(split, category)] += 1
                    continue
                valid_count += 1
                tactic = str(trace.get("tactic", ""))
                tactic_name = _tactic_name(tactic)
                try:
                    target = compile_action_trace(trace)
                except ActionTargetError as exc:
                    failure_categories[exc.code] += 1
                    per_tactic[tactic_name][f"failure:{exc.code}"] += 1
                    if sample_counts[(split, exc.code)] < config.sample_limit_per_category:
                        samples_handle.write(
                            json.dumps(
                                {
                                    "split": split,
                                    "row_index": row_index,
                                    "theorem": trace.get("theorem"),
                                    "tactic": tactic,
                                    "category": exc.code,
                                    "error": str(exc),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        sample_counts[(split, exc.code)] += 1
                    continue

                compiled_count += 1
                per_tactic[tactic_name]["compiled"] += 1
                if target.has_payload:
                    payload_count += 1
                    per_tactic[tactic_name]["payload"] += 1
                else:
                    empty_count += 1
                    per_tactic[tactic_name]["empty_payload"] += 1
                    category = "empty_payload"
                    if sample_counts[(split, category)] < config.sample_limit_per_category:
                        samples_handle.write(
                            json.dumps(
                                {
                                    "split": split,
                                    "row_index": row_index,
                                    "theorem": trace.get("theorem"),
                                    "tactic": tactic,
                                    "category": category,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        sample_counts[(split, category)] += 1

                for operation in target.operations:
                    operation_counts[str(operation["op"])] += 1
                sequence_lengths.append(len(target.operations))
                targets_handle.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "split": split,
                            "row_index": row_index,
                            "theorem": trace.get("theorem"),
                            "tactic": tactic,
                            "term_count": target.term_count,
                            "syntax_argument_count": target.syntax_argument_count,
                            "operations": target.operations,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                if processed == 1 or processed % 10000 == 0:
                    console_print(f"  [{split}] compiled {processed}/{len(trace_files)} traces")

            split_summary = {
                "trace_file_count": len(trace_files),
                "valid_trace_count": valid_count,
                "compiled_count": compiled_count,
                "failure_count": len(trace_files) - compiled_count,
                "payload_row_count": payload_count,
                "empty_payload_row_count": empty_count,
                "compiler_success_rate": compiled_count / max(len(trace_files), 1),
                "payload_rate": payload_count / max(compiled_count, 1),
                "operation_counts": dict(sorted(operation_counts.items())),
                "failure_categories": dict(sorted(failure_categories.items())),
                "sequence_lengths": {
                    "median": _percentile(sequence_lengths, 0.5),
                    "p95": _percentile(sequence_lengths, 0.95),
                    "p99": _percentile(sequence_lengths, 0.99),
                    "max": max(sequence_lengths, default=0),
                },
                "extraction_manifest": _manifest_counts(config.cache_root, split),
                "per_tactic": {
                    name: dict(sorted(counts.items()))
                    for name, counts in sorted(per_tactic.items())
                },
            }
            summary_splits = summary["splits"]
            if not isinstance(summary_splits, dict):
                raise TypeError("Internal split summary is invalid.")
            summary_splits[split] = split_summary
            console_print(
                f"  [{split}] compiled={compiled_count}/{len(trace_files)} "
                f"payload={payload_count} empty={empty_count}"
            )

    summary_path = config.output_dir / "summary.json"
    markdown_path = config.output_dir / "summary.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    console_print(f"  Wrote summary : {summary_path}")
    console_print(f"  Wrote targets : {targets_path}")
    console_print(f"  Wrote samples : {samples_path}")
    return summary
