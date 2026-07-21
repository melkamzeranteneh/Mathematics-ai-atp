#!/usr/bin/env python3
"""Extract validated Lean S-expressions by replaying complete theorems."""

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
    SExprExtractionConfig,
    extract_sexpressions,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay each LeanDojo theorem from its declaration and cache the "
            "S-expression state immediately before every dataset tactic."
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
    parser.add_argument("--project-path", default="maths_ai/lean_mathlib")
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=600,
        help="Seconds allowed for Pantograph to import Mathlib and become ready.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Replay every theorem even when all of its versioned rows exist.",
    )
    parser.add_argument(
        "--allow-incomplete-theorems",
        action="store_true",
        help="Do not require the final replayed tactic to close every goal.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = SExprExtractionConfig(
        prepared_root=args.prepared_root,
        dataset_name=args.dataset_name,
        splits=tuple(args.splits),
        project_path=args.project_path,
        sample_per_split=args.max_items,
        resume=not args.no_resume,
        require_solved_theorem=not args.allow_incomplete_theorems,
        server_startup_timeout=args.server_timeout,
    )
    try:
        summary = asyncio.run(extract_sexpressions(config))
    except KeyboardInterrupt:
        print("Interrupted; completed theorem caches remain valid and resumable.")
        return 130
    except Exception as exc:
        print(f"S-expression extraction failed: {exc}")
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if int(summary["failed_rows"]) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
