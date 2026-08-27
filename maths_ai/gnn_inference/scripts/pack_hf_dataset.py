#!/usr/bin/env python3
"""Pack the validated S-expression caches into publishable Parquet shards.

The extraction caches store one JSON file per row per target, which is the right
shape for resumable extraction and the wrong shape for distribution: 227k rows
across three targets is roughly 680k files, a count that degrades both git-LFS
and the Hugging Face Hub.

This packer joins the three caches into a single self-contained row and writes
sharded Parquet.  Joining matters for consumers: a ``model_sexpr_v2`` sidecar is
only valid while its ``raw_record_sha256`` matches the raw record it was derived
from, so the sidecars are not independently loadable.  Emitting one row that
carries the raw state, the normalized state, and the action trace together
dissolves that coupling at publish time, and the digest is retained as a column
so the binding stays auditable.

Nested payloads are stored as JSON strings rather than nested Parquet types.
The ``source_syntax`` tree is recursive and its shape varies per tactic, so a
typed column would either fail schema inference or force a lowest-common-
denominator union.  A JSON string keeps every shard's schema identical, which is
what lets ``load_dataset`` read the whole split without hints.

Only rows whose three targets are all present and digest-valid are emitted, so
the packed row count equals the coverage reported by the extraction manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "pack_hf_dataset requires pyarrow. Install it with 'pip install pyarrow'."
    ) from exc


# These mirror the cache classes in atp_lean_gnn/preparation.py. They are
# duplicated rather than imported because importing that package pulls in torch
# by way of atp_lean_gnn/__init__.py, and packing JSON into Parquet has no
# business requiring a deep-learning stack. _assert_no_schema_drift() below
# re-checks every value against the real definitions whenever they do import, so
# a version bump upstream fails loudly here instead of silently publishing rows
# under a stale version label.
CANONICAL_SPLITS = ("train", "val", "test")
RAW_DIR = "sexpr"
RAW_SCHEMA_VERSION = 4
MODEL_DIR = "model_sexpr_v2"
MODEL_SCHEMA_VERSION = 2
TRACE_DIR = "action_trace_v3"
TRACE_SCHEMA_VERSION = 3


def raw_record_sha256(raw_record: dict) -> str:
    """Digest binding a sidecar to the raw record it was derived from.

    Byte-identical to ModelSExprCache.raw_record_sha256; any divergence would
    reject every sidecar as mismatched.
    """
    encoded = json.dumps(
        raw_record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_no_schema_drift() -> None:
    """Fail loudly if the mirrored constants no longer match preparation.py."""
    if __package__ in {None, ""}:
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    try:
        from maths_ai.gnn_inference.atp_lean_gnn.dataset import CANONICAL_SPLITS as real_splits
        from maths_ai.gnn_inference.atp_lean_gnn.preparation import (
            ModelSExprCache,
            SExprCache,
            SourceSyntaxTraceCache,
        )
    except Exception:
        # torch or another optional dependency is absent; the mirrored values
        # stand on their own and packing proceeds.
        return

    probe = {"b": 2, "a": "é"}
    mismatches = [
        name
        for name, mirrored, actual in (
            ("CANONICAL_SPLITS", CANONICAL_SPLITS, tuple(real_splits)),
            ("RAW_SCHEMA_VERSION", RAW_SCHEMA_VERSION, SExprCache.SCHEMA_VERSION),
            ("MODEL_SCHEMA_VERSION", MODEL_SCHEMA_VERSION, ModelSExprCache.SCHEMA_VERSION),
            ("MODEL_DIR", MODEL_DIR, "model_sexpr_v2"),
            ("TRACE_DIR", TRACE_DIR, SourceSyntaxTraceCache.SIDECAR_DIR),
            ("TRACE_SCHEMA_VERSION", TRACE_SCHEMA_VERSION, SourceSyntaxTraceCache.SCHEMA_VERSION),
            ("raw_record_sha256", raw_record_sha256(probe), ModelSExprCache.raw_record_sha256(probe)),
        )
        if mirrored != actual
    ]
    if mismatches:
        raise SystemExit(
            "pack_hf_dataset is out of sync with preparation.py: "
            + ", ".join(mismatches)
        )


SCHEMA = pa.schema(
    [
        # Row identity, sufficient to rejoin against the upstream dataset.
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("row_index", pa.int64()),
        ("theorem", pa.string()),
        ("file_path", pa.string()),
        ("repo_url", pa.string()),
        ("repo_commit", pa.string()),
        # Original supervision text, as observed by Lean at the invocation.
        ("tactic", pa.string()),
        ("text_state", pa.string()),
        ("text_target_state", pa.string()),
        # Source-faithful S-expression state.
        ("raw_goal_sexp", pa.string()),
        ("raw_hyp_sexps", pa.string()),
        # Normalized model-facing state.
        ("model_goal_sexp", pa.string()),
        ("model_hyp_sexps", pa.string()),
        # Decoder target: annotated original tactic syntax.
        ("source_syntax", pa.string()),
        ("syntax_args", pa.string()),
        ("term_ranges", pa.string()),
        ("local_context", pa.string()),
        # Alignment diagnostics from the invocation match.
        ("unit_index", pa.int64()),
        ("invocation_index", pa.int64()),
        ("alignment_kind", pa.string()),
        ("target_state_matches_invocation", pa.bool_()),
        ("pending_goal_count", pa.int64()),
        ("hypothesis_count", pa.int64()),
        ("hypothesis_names_match", pa.bool_()),
        # Provenance, so a row can be traced back to the exact extractor.
        ("raw_schema_version", pa.int64()),
        ("raw_extractor_version", pa.string()),
        ("model_schema_version", pa.int64()),
        ("model_normalization", pa.string()),
        ("trace_schema_version", pa.int64()),
        ("trace_extractor_version", pa.string()),
        ("pantograph_commit", pa.string()),
        ("model_pantograph_commit", pa.string()),
        ("raw_record_sha256", pa.string()),
        ("state_sha256", pa.string()),
        ("tactic_sha256", pa.string()),
        ("target_state_sha256", pa.string()),
    ]
)


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _row_indices(directory: Path) -> list[int]:
    """Row indices present in a cache directory, in ascending order."""
    indices = []
    for path in directory.glob("*.json"):
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    return sorted(indices)


def _build_row(
    raw: dict,
    model: dict,
    trace: dict,
) -> dict[str, object]:
    return {
        "dataset": raw.get("dataset"),
        "split": raw.get("split"),
        "row_index": raw.get("row_index"),
        "theorem": raw.get("theorem"),
        "file_path": raw.get("file_path"),
        "repo_url": raw.get("repo_url"),
        "repo_commit": raw.get("repo_commit"),
        "tactic": raw.get("tactic"),
        "text_state": raw.get("text_state"),
        "text_target_state": raw.get("text_target_state"),
        "raw_goal_sexp": raw.get("goal_sexp"),
        "raw_hyp_sexps": _dumps(raw.get("hyp_sexps", [])),
        "model_goal_sexp": model.get("goal_sexp"),
        "model_hyp_sexps": _dumps(model.get("hyp_sexps", [])),
        "source_syntax": _dumps(trace.get("source_syntax")),
        "syntax_args": _dumps(trace.get("syntax_args", [])),
        "term_ranges": _dumps(trace.get("term_ranges", [])),
        "local_context": _dumps(trace.get("local_context", [])),
        "unit_index": raw.get("unit_index"),
        "invocation_index": raw.get("invocation_index"),
        "alignment_kind": raw.get("alignment_kind"),
        "target_state_matches_invocation": raw.get("target_state_matches_invocation"),
        "pending_goal_count": raw.get("pending_goal_count"),
        "hypothesis_count": raw.get("hypothesis_count"),
        "hypothesis_names_match": raw.get("hypothesis_names_match"),
        "raw_schema_version": raw.get("schema_version"),
        "raw_extractor_version": raw.get("extractor_version"),
        "model_schema_version": model.get("schema_version"),
        "model_normalization": model.get("normalization"),
        "trace_schema_version": trace.get("schema_version"),
        "trace_extractor_version": trace.get("extractor_version"),
        "pantograph_commit": raw.get("pantograph_commit"),
        "model_pantograph_commit": model.get("pantograph_commit"),
        "raw_record_sha256": trace.get("raw_record_sha256"),
        "state_sha256": raw.get("state_sha256"),
        "tactic_sha256": raw.get("tactic_sha256"),
        "target_state_sha256": raw.get("target_state_sha256"),
    }


def _iter_split_rows(
    prepared_root: Path,
    split: str,
    stats: dict[str, int],
) -> Iterator[dict[str, object]]:
    """Yield one joined row per fully valid triple, counting what is skipped.

    The version-3 trace directory gates iteration because it is the smallest of
    the three and the last one written, so every candidate row necessarily has a
    trace.  A missing or digest-mismatched counterpart is skipped rather than
    raising: partially written rows are an expected consequence of interrupting
    extraction, and dropping them is what keeps the packed count equal to the
    manifest's coverage.
    """
    raw_dir = prepared_root / split / RAW_DIR
    model_dir = prepared_root / split / MODEL_DIR
    trace_dir = prepared_root / split / TRACE_DIR

    if not trace_dir.is_dir():
        return

    for row_index in _row_indices(trace_dir):
        name = f"{row_index:09d}.json"
        trace = _load_json(trace_dir / name)
        raw = _load_json(raw_dir / name)
        model = _load_json(model_dir / name)

        if trace is None or raw is None or model is None:
            stats["missing_target"] += 1
            continue
        if (
            raw.get("schema_version") != RAW_SCHEMA_VERSION
            or model.get("schema_version") != MODEL_SCHEMA_VERSION
            or trace.get("schema_version") != TRACE_SCHEMA_VERSION
        ):
            stats["schema_mismatch"] += 1
            continue

        # Both sidecars are bound to the raw record by digest; a mismatch means
        # the raw record was re-extracted after the sidecar was written, so the
        # pair no longer describes the same proof state.
        digest = raw_record_sha256(raw)
        if (
            model.get("raw_record_sha256") != digest
            or trace.get("raw_record_sha256") != digest
        ):
            stats["digest_mismatch"] += 1
            continue

        stats["emitted"] += 1
        yield _build_row(raw, model, trace)


def _completed_shards(
    split_root: Path, split: str, shard_rows: int
) -> tuple[list[Path], int]:
    """Existing full shards for a split, and the row count they already cover.

    Packing a large split is a single long pass, so an interrupted run must not
    force the whole split to be redone. Shards are only ever appended, and
    ``_iter_split_rows`` walks row indices in sorted order, so a shard holding
    exactly ``shard_rows`` rows is a complete, reusable prefix. A trailing short
    shard is by definition the one that was being written when the run died: its
    contents are a prefix of what the next shard should hold, but it cannot be
    appended to, so it is discarded and rebuilt.
    """
    shards = sorted(split_root.glob(f"{split}-*.parquet"))
    if not shards:
        return [], 0

    complete: list[Path] = []
    for index, path in enumerate(shards):
        try:
            rows = pq.ParquetFile(path).metadata.num_rows
        except Exception:
            # Unreadable, most likely truncated mid-write.
            rows = -1
        if rows == shard_rows:
            complete.append(path)
            continue
        if index != len(shards) - 1:
            raise SystemExit(
                f"{path.name} holds {rows} rows, expected {shard_rows}, and it "
                "is not the last shard. Only the final shard may be short, so "
                "this set was written with a different --shard-rows or is "
                f"corrupt; delete {split_root} and pack this split again."
            )
        # The final shard is either short because the split ended there or
        # truncated because the run was killed mid-write. The two are
        # indistinguishable from disk, so it is always discarded and rebuilt.
        path.unlink()

    for path in shards[len(complete) :]:
        path.unlink(missing_ok=True)
    return complete, len(complete) * shard_rows


def pack_split(
    prepared_root: Path,
    output_root: Path,
    split: str,
    *,
    shard_rows: int,
    batch_rows: int,
    compression: str,
    resume: bool,
) -> dict[str, object]:
    stats = {
        "emitted": 0,
        "missing_target": 0,
        "schema_mismatch": 0,
        "digest_mismatch": 0,
    }
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)

    if resume:
        shard_paths, resumed_rows = _completed_shards(split_root, split, shard_rows)
    else:
        for stale in split_root.glob(f"{split}-*.parquet"):
            stale.unlink()
        shard_paths, resumed_rows = [], 0
    if resumed_rows:
        print(
            f"{split}: resuming after {resumed_rows} rows in "
            f"{len(shard_paths)} complete shards",
            flush=True,
        )

    writer: pq.ParquetWriter | None = None
    rows_in_shard = 0
    batch: list[dict[str, object]] = []

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=SCHEMA)
        assert writer is not None
        writer.write_table(table)
        batch = []

    def close_shard() -> None:
        nonlocal writer, rows_in_shard
        if writer is None:
            return
        flush_batch()
        writer.close()
        writer = None
        rows_in_shard = 0

    partial_path: Path | None = None
    try:
        for row in _iter_split_rows(prepared_root, split, stats):
            # Rows already covered by complete shards are re-read but not
            # rewritten. Re-reading them is the price of not tracking a
            # separate cursor that could disagree with the shards on disk.
            if stats["emitted"] <= resumed_rows:
                continue
            if writer is None:
                partial_path = split_root / f"{split}-{len(shard_paths):05d}.parquet"
                shard_paths.append(partial_path)
                writer = pq.ParquetWriter(
                    partial_path, SCHEMA, compression=compression
                )
            batch.append(row)
            rows_in_shard += 1
            if len(batch) >= batch_rows:
                flush_batch()
            if rows_in_shard >= shard_rows:
                close_shard()
                partial_path = None
            if stats["emitted"] % 25_000 == 0:
                print(f"{split}: packed {stats['emitted']} rows", flush=True)
        close_shard()
        partial_path = None
    except BaseException:
        # Leave only complete shards behind. A truncated shard would otherwise
        # be counted as complete on the next resume and silently drop rows.
        if writer is not None:
            writer.close()
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
        raise

    shard_names = sorted(path.name for path in split_root.glob(f"{split}-*.parquet"))
    bytes_written = sum(
        (split_root / name).stat().st_size for name in shard_names
    )
    print(
        f"{split}: {stats['emitted']} rows in {len(shard_names)} shards "
        f"({bytes_written / 1e9:.2f} GB)",
        flush=True,
    )
    return {
        "split": split,
        "rows": stats["emitted"],
        "shards": shard_names,
        "bytes": bytes_written,
        "skipped": {
            key: value for key, value in stats.items() if key != "emitted"
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join the raw, normalized, and action-trace caches into sharded "
            "Parquet suitable for publishing to the Hugging Face Hub. Only rows "
            "whose three targets are all present and digest-valid are emitted."
        )
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="Prepared root holding the per-split extraction caches.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Destination for the Parquet shards. Upload this directory and "
            "nothing else; pointing an upload at the repository root would "
            "publish the Mathlib checkout and the raw caches as well."
        ),
    )
    parser.add_argument("--splits", nargs="+", default=list(CANONICAL_SPLITS))
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=5000,
        help=(
            "Rows per Parquet shard. The default keeps each shard a few hundred "
            "megabytes, which is the range the Hub streams comfortably."
        ),
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=500,
        help=(
            "Rows buffered in memory before each Arrow write. Bounds peak "
            "memory, which matters when packing runs beside other jobs."
        ),
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
        help="Parquet codec. S-expression text is repetitive, so zstd wins.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Repack from scratch, discarding existing shards. By default an "
            "interrupted run resumes from its complete shards, which matters "
            "because packing a large split is a single long pass."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _assert_no_schema_drift()
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve()
    if not prepared_root.is_dir():
        print(f"Prepared root does not exist: {prepared_root}")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in args.splits:
        summaries.append(
            pack_split(
                prepared_root,
                output_root,
                split,
                shard_rows=args.shard_rows,
                batch_rows=args.batch_rows,
                compression=args.compression,
                resume=not args.no_resume,
            )
        )

    report = {
        "prepared_root": str(prepared_root),
        "output_root": str(output_root),
        "compression": args.compression,
        "shard_rows": args.shard_rows,
        "schema_fields": [field.name for field in SCHEMA],
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "splits": {summary["split"]: summary for summary in summaries},
        "total_rows": sum(int(summary["rows"]) for summary in summaries),
        "total_bytes": sum(int(summary["bytes"]) for summary in summaries),
    }
    report_path = output_root / "pack_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["total_rows"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
