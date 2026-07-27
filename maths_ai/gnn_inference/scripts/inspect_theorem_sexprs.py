#!/usr/bin/env python3
"""Inspect every cached S-expression proof state for one dataset theorem."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from maths_ai.gnn_inference.atp_lean_gnn.dataset import (
    DATASET_NAME,
    canonicalize_split_name,
    iter_dataset_rows,
)
from maths_ai.gnn_inference.atp_lean_gnn.preparation import SExprCache
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_inspection import (
    build_theorem_trace,
    write_theorem_trace,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write Markdown and JSON traces containing every dataset proof state, "
            "tactic, target state, goal S-expression, and hypothesis S-expression "
            "for one theorem."
        )
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--theorem", required=True, help="Exact dataset theorem name.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: PREPARED_ROOT/sexpr_inspection",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    split = canonicalize_split_name(args.split)
    cache = SExprCache(args.prepared_root, project_path="", enabled=True)
    try:
        trace = build_theorem_trace(
            rows=iter_dataset_rows(
                dataset_name=args.dataset_name,
                split=split,
            ),
            theorem=args.theorem,
            split=split,
            cache=cache,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"S-expression inspection failed: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or args.prepared_root / "sexpr_inspection"
    json_path, markdown_path = write_theorem_trace(trace, output_dir=output_dir)
    print(f"Theorem                 : {trace['theorem']}")
    print(f"Dataset rows            : {trace['dataset_row_count']}")
    print(f"Validated S-expression  : {trace['cached_row_count']}")
    print(f"Complete                : {'yes' if trace['complete'] else 'no'}")
    print(f"JSON trace              : {json_path}")
    print(f"Markdown trace          : {markdown_path}")
    if not trace["complete"]:
        print(f"Missing rows            : {trace['missing_row_indices']}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
