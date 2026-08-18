from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..dataset import canonicalize_split_name
from ..preparation import ActionTraceCache, SExprCache, SourceSyntaxTraceCache
from ..reporting import console_print
from .compiler import ActionTargetError, Operation, compile_action_trace
from .source_syntax import compile_source_syntax_trace


TRACE_VERSIONS = ("v2", "v3")

_TRACE_CACHES: dict[str, type[ActionTraceCache]] = {
    "v2": ActionTraceCache,
    "v3": SourceSyntaxTraceCache,
}


@dataclass(frozen=True)
class ActionTargetAuditConfig:
    cache_root: Path
    output_dir: Path
    splits: tuple[str, ...] = ("train",)
    trace_version: str = "v3"
    max_operations: int | None = None
    max_items_per_split: int | None = None
    sample_limit_per_category: int = 20
    force: bool = False

    def normalized(self) -> "ActionTargetAuditConfig":
        splits = tuple(canonicalize_split_name(split) for split in self.splits)
        if not splits:
            raise ValueError("At least one split is required.")
        if self.trace_version not in _TRACE_CACHES:
            raise ValueError(
                f"Unknown trace version {self.trace_version!r}; "
                f"expected one of {TRACE_VERSIONS}."
            )
        if self.max_operations is not None and self.max_operations < 1:
            raise ValueError("max_operations must be positive when provided.")
        if self.max_items_per_split is not None and self.max_items_per_split < 1:
            raise ValueError("max_items_per_split must be positive when provided.")
        if self.sample_limit_per_category < 0:
            raise ValueError("sample_limit_per_category cannot be negative.")
        return ActionTargetAuditConfig(
            cache_root=self.cache_root.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            splits=splits,
            trace_version=self.trace_version,
            max_operations=self.max_operations,
            max_items_per_split=self.max_items_per_split,
            sample_limit_per_category=self.sample_limit_per_category,
            force=self.force,
        )

    @property
    def trace_cache_class(self) -> type[ActionTraceCache]:
        return _TRACE_CACHES[self.trace_version]


@dataclass(frozen=True)
class _AuditedTarget:
    """Version-independent view of one compiled target."""

    operations: tuple[Operation, ...]
    has_payload: bool
    metrics: dict[str, int] = field(default_factory=dict)


def _audit_v2_target(trace: dict) -> _AuditedTarget:
    target = compile_action_trace(trace)
    return _AuditedTarget(
        operations=target.operations,
        has_payload=target.has_payload,
        metrics={
            "terms": target.term_count,
            "syntax_arguments": target.syntax_argument_count,
        },
    )


def _audit_v3_target(trace: dict) -> _AuditedTarget:
    target = compile_source_syntax_trace(trace)
    return _AuditedTarget(
        operations=target.operations,
        has_payload=target.has_payload,
        metrics={
            "nodes": target.node_count,
            "atoms": target.atom_count,
            "empty_null_nodes": target.empty_null_node_count,
            "local_references": target.local_reference_count,
            "scoped_locals": target.scoped_local_count,
            "unannotated_identifiers": target.unannotated_identifier_count,
            "fresh_names": target.fresh_name_count,
            "missing_syntax": target.missing_count,
        },
    )


_TRACE_COMPILERS = {"v2": _audit_v2_target, "v3": _audit_v3_target}


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _tactic_name(tactic: str) -> str:
    parts = tactic.strip().split(maxsplit=1)
    return parts[0] if parts else "<empty>"


def _manifest_counts(
    cache_root: Path, split: str, report_root: str
) -> dict[str, int | float] | None:
    path = cache_root / report_root / "manifests" / f"{split}.json"
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
        f"- Trace version: `{summary['trace_version']}`",
        f"- Trace extractor: `{summary['trace_extractor_version']}`",
        f"- Operation cap: `{summary['max_operations']}`",
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
                f"- Accepted targets: `{payload['accepted_count']}`",
                f"- Over operation cap: `{payload['over_cap_count']}`",
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
                f"- over 256: `{payload['sequence_lengths']['over_256']}`",
                f"- over 512: `{payload['sequence_lengths']['over_512']}`",
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
            [
                "",
                "### Target content",
                "",
                "| Metric | Total | Targets |",
                "| --- | ---: | ---: |",
            ]
        )
        metric_totals = dict(payload["metric_totals"])
        targets_with_metric = dict(payload["targets_with_metric"])
        for metric, total in metric_totals.items():
            lines.append(
                f"| {metric} | {total} | {targets_with_metric.get(metric, 0)} |"
            )
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

    cache_class = config.trace_cache_class
    compile_target = _TRACE_COMPILERS[config.trace_version]
    raw_cache = SExprCache(config.cache_root, project_path="", enabled=True)
    action_cache = cache_class(config.cache_root, enabled=True)
    targets_path = config.output_dir / "targets.jsonl"
    samples_path = config.output_dir / "samples.jsonl"
    summary: dict[str, object] = {
        "schema_version": 2,
        "cache_root": str(config.cache_root),
        "trace_version": config.trace_version,
        "trace_extractor_version": cache_class.EXTRACTOR_VERSION,
        "max_operations": config.max_operations,
        "splits": {},
    }
    sample_counts: Counter[tuple[str, str]] = Counter()

    with targets_path.open("w", encoding="utf-8") as targets_handle, samples_path.open(
        "w", encoding="utf-8"
    ) as samples_handle:
        for split in config.splits:
            trace_dir = config.cache_root / split / cache_class.SIDECAR_DIR
            trace_files = sorted(trace_dir.glob("*.json")) if trace_dir.exists() else []
            if config.max_items_per_split is not None:
                trace_files = trace_files[: config.max_items_per_split]
            if not trace_files:
                raise FileNotFoundError(
                    f"No {cache_class.SIDECAR_DIR} sidecars found under {trace_dir}"
                )

            valid_count = compiled_count = payload_count = empty_count = 0
            accepted_count = over_cap_count = 0
            failure_categories: Counter[str] = Counter()
            operation_counts: Counter[str] = Counter()
            metric_totals: Counter[str] = Counter()
            targets_with_metric: Counter[str] = Counter()
            sequence_lengths: list[int] = []
            per_tactic: defaultdict[str, Counter[str]] = defaultdict(Counter)

            def record_sample(category: str, payload: dict[str, object]) -> None:
                if sample_counts[(split, category)] >= config.sample_limit_per_category:
                    return
                samples_handle.write(
                    json.dumps(
                        {"split": split, "category": category, **payload},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                sample_counts[(split, category)] += 1

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
                    failure_categories["invalid_or_stale_trace"] += 1
                    record_sample("invalid_or_stale_trace", {"row_index": row_index})
                    continue
                valid_count += 1
                tactic = str(trace.get("tactic", ""))
                tactic_name = _tactic_name(tactic)
                try:
                    target = compile_target(trace)
                except ActionTargetError as exc:
                    failure_categories[exc.code] += 1
                    per_tactic[tactic_name][f"failure:{exc.code}"] += 1
                    record_sample(
                        exc.code,
                        {
                            "row_index": row_index,
                            "theorem": trace.get("theorem"),
                            "tactic": tactic,
                            "error": str(exc),
                        },
                    )
                    continue

                compiled_count += 1
                per_tactic[tactic_name]["compiled"] += 1
                if target.has_payload:
                    payload_count += 1
                    per_tactic[tactic_name]["payload"] += 1
                else:
                    empty_count += 1
                    per_tactic[tactic_name]["empty_payload"] += 1
                    record_sample(
                        "empty_payload",
                        {
                            "row_index": row_index,
                            "theorem": trace.get("theorem"),
                            "tactic": tactic,
                        },
                    )

                for operation in target.operations:
                    operation_counts[str(operation["op"])] += 1
                for metric, value in target.metrics.items():
                    metric_totals[metric] += value
                    if value > 0:
                        targets_with_metric[metric] += 1
                operation_count = len(target.operations)
                sequence_lengths.append(operation_count)

                if (
                    config.max_operations is not None
                    and operation_count > config.max_operations
                ):
                    # Kept out of the training targets but still counted, so the
                    # cap's cost stays visible instead of looking like coverage.
                    over_cap_count += 1
                    per_tactic[tactic_name]["over_cap"] += 1
                    record_sample(
                        "target_too_long",
                        {
                            "row_index": row_index,
                            "theorem": trace.get("theorem"),
                            "tactic": tactic,
                            "operation_count": operation_count,
                        },
                    )
                    continue

                accepted_count += 1
                targets_handle.write(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "trace_version": config.trace_version,
                            "split": split,
                            "row_index": row_index,
                            "theorem": trace.get("theorem"),
                            "tactic": tactic,
                            "metrics": target.metrics,
                            "operations": target.operations,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                if processed == 1 or processed % 10000 == 0:
                    console_print(
                        f"  [{split}] compiled {processed}/{len(trace_files)} traces"
                    )

            split_summary = {
                "trace_file_count": len(trace_files),
                "valid_trace_count": valid_count,
                "compiled_count": compiled_count,
                "accepted_count": accepted_count,
                "over_cap_count": over_cap_count,
                "failure_count": len(trace_files) - compiled_count,
                "payload_row_count": payload_count,
                "empty_payload_row_count": empty_count,
                "compiler_success_rate": compiled_count / max(len(trace_files), 1),
                "accepted_rate": accepted_count / max(len(trace_files), 1),
                "payload_rate": payload_count / max(compiled_count, 1),
                "operation_counts": dict(sorted(operation_counts.items())),
                "metric_totals": dict(sorted(metric_totals.items())),
                "targets_with_metric": dict(sorted(targets_with_metric.items())),
                "failure_categories": dict(sorted(failure_categories.items())),
                "sequence_lengths": {
                    "median": _percentile(sequence_lengths, 0.5),
                    "p95": _percentile(sequence_lengths, 0.95),
                    "p99": _percentile(sequence_lengths, 0.99),
                    "max": max(sequence_lengths, default=0),
                    "over_256": sum(1 for length in sequence_lengths if length > 256),
                    "over_512": sum(1 for length in sequence_lengths if length > 512),
                },
                "extraction_manifest": _manifest_counts(
                    config.cache_root, split, cache_class.REPORT_ROOT
                ),
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
                f"accepted={accepted_count} payload={payload_count} empty={empty_count}"
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
