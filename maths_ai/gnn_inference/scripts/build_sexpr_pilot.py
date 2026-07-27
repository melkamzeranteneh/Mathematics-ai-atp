#!/usr/bin/env python3
"""Build a deterministic theorem-level selection for raw/model graph ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DATASET_NAME
from maths_ai.gnn_inference.atp_lean_gnn.pilot_sampling import (
    build_pilot_selection,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--train-rows", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-train-raw-coverage", type=float, default=0.8)
    parser.add_argument(
        "--allow-incomplete-eval",
        action="store_true",
        help="Select only raw-covered validation/test rows instead of failing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = build_pilot_selection(
            prepared_root=args.prepared_root,
            output_path=args.output,
            dataset_name=args.dataset_name,
            target_train_rows=args.train_rows,
            seed=args.seed,
            minimum_train_raw_coverage=args.minimum_train_raw_coverage,
            require_complete_eval=not args.allow_incomplete_eval,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Pilot selection failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "output": str(args.output),
        "selected_train_rows": manifest["selected_train_rows"],
        "train_raw_coverage": manifest["train_raw_coverage"],
        "stratum_count": manifest["stratum_count"],
        "train_tactic_total_variation": manifest["splits"]["train"][
            "tactic_total_variation"
        ],
        "validation_rows": manifest["splits"]["val"]["row_count"],
        "test_rows": manifest["splits"]["test"]["row_count"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
