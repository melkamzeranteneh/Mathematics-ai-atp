from __future__ import annotations

import argparse
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.action_targets import (
    ActionTargetAuditConfig,
    run_action_target_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and audit Lean-native action_trace_v2 decoder targets."
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", type=str, default="train")
    parser.add_argument("--max-items-per-split", type=int, default=None)
    parser.add_argument("--sample-limit-per-category", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_root = args.cache_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else cache_root / "action_trace_extraction_v2" / "target_audit"
    )
    try:
        run_action_target_audit(
            ActionTargetAuditConfig(
                cache_root=cache_root,
                output_dir=output_dir,
                splits=tuple(
                    split.strip() for split in args.splits.split(",") if split.strip()
                ),
                max_items_per_split=args.max_items_per_split,
                sample_limit_per_category=args.sample_limit_per_category,
                force=args.force,
            )
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Structured action target audit failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
