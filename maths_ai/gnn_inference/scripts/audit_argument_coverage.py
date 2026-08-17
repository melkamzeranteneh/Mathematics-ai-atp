from __future__ import annotations

import argparse
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.argument_coverage import (
    ArgumentCoverageConfig,
    run_argument_coverage_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify why prepared pointer argument targets are or are not selectable."
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--lemma-corpus", type=Path, default=None)
    parser.add_argument(
        "--lemma-index",
        type=Path,
        default=None,
        help="Lemma index directory containing lemma_names.json, or the names JSON itself.",
    )
    parser.add_argument("--max-items-per-split", type=int, default=None)
    parser.add_argument("--sample-limit-per-category", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared_root = args.prepared_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else prepared_root / "reports" / "argument_coverage"
    )
    try:
        run_argument_coverage_audit(
            ArgumentCoverageConfig(
                prepared_root=prepared_root,
                output_dir=output_dir,
                splits=tuple(
                    split.strip() for split in args.splits.split(",") if split.strip()
                ),
                lemma_corpus=args.lemma_corpus,
                lemma_index=args.lemma_index,
                max_items_per_split=args.max_items_per_split,
                sample_limit_per_category=args.sample_limit_per_category,
                force=args.force,
            )
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Argument coverage audit failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
