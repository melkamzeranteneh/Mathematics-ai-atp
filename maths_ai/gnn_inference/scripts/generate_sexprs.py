#!/usr/bin/env python3
"""Extract S-expressions from original, dataset-version Mathlib sources."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DATASET_NAME
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import (
    MODEL_PANTOGRAPH_COMMIT,
    PANTOGRAPH_COMMIT,
    SExprExtractionConfig,
    extract_sexpressions,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile each original Mathlib file and cache the authentic "
            "S-expression state recorded at every matched tactic invocation."
        )
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="Destination root for versioned row caches and extraction reports.",
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help=(
            "Exact theorem-level pilot selection produced by build_sexpr_pilot. "
            "Cannot be combined with --max-items."
        ),
    )
    parser.add_argument(
        "--theorem",
        default=None,
        help="Extract only rows belonging to this exact theorem name.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Mathlib checkout at the exact commit stored in the dataset.",
    )
    parser.add_argument(
        "--pantograph-repl",
        type=Path,
        required=True,
        help="Patched Lean-4.10 Pantograph REPL built by setup_sexpr_environment.",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=600,
        help="Seconds allowed for Pantograph to become ready.",
    )
    parser.add_argument(
        "--file-timeout",
        type=int,
        default=600,
        help="Seconds allowed to compile one original Mathlib source file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of independent Pantograph REPL workers. Each worker compiles "
            "one source file at a time; start conservatively with 3."
        ),
    )
    parser.add_argument(
        "--recycle-worker-files",
        type=int,
        default=10,
        help=(
            "Restart each Pantograph REPL after this many files to release Lean "
            "frontend memory. Use 0 to disable recycling."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-extract rows even when their validated versioned cache exists.",
    )
    parser.add_argument(
        "--model-sexprs",
        action="store_true",
        help=(
            "Compile with the fork's Lean-native normalizer and write versioned "
            "model_sexpr_v2 sidecars. Existing validated raw caches are required "
            "and are never overwritten."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = SExprExtractionConfig(
        prepared_root=args.prepared_root,
        source_root=args.source_root,
        pantograph_repl=args.pantograph_repl,
        dataset_name=args.dataset_name,
        splits=tuple(args.splits),
        sample_per_split=args.max_items,
        resume=not args.no_resume,
        server_startup_timeout=args.server_timeout,
        file_timeout=args.file_timeout,
        workers=args.workers,
        recycle_worker_files=args.recycle_worker_files,
        model_sexprs=args.model_sexprs,
        expected_pantograph_commit=(
            MODEL_PANTOGRAPH_COMMIT if args.model_sexprs else PANTOGRAPH_COMMIT
        ),
        theorem=args.theorem,
        selection_manifest=args.selection_manifest,
    )
    try:
        summary = asyncio.run(extract_sexpressions(config))
    except KeyboardInterrupt:
        print("Interrupted; completed row caches remain valid and resumable.")
        return 130
    except Exception as exc:
        print(f"S-expression extraction failed: {exc}")
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if int(summary["failed_rows"]) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
