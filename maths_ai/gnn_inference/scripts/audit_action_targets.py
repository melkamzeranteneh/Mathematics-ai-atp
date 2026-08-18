from __future__ import annotations

import argparse
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.action_targets import (
    TRACE_VERSIONS,
    ActionTargetAuditConfig,
    run_action_target_audit,
)
from maths_ai.gnn_inference.atp_lean_gnn.preparation import (
    ActionTraceCache,
    SourceSyntaxTraceCache,
)


REPORT_ROOTS = {
    "v2": ActionTraceCache.REPORT_ROOT,
    "v3": SourceSyntaxTraceCache.REPORT_ROOT,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and audit Lean-native action-trace decoder targets."
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", type=str, default="train")
    parser.add_argument(
        "--trace-version",
        choices=TRACE_VERSIONS,
        default="v3",
        help=(
            "v3 compiles the compact annotated tactic syntax; v2 compiles the "
            "older fully elaborated tactic terms."
        ),
    )
    parser.add_argument(
        "--max-operations",
        type=int,
        default=None,
        help=(
            "Exclude compiled targets longer than this from targets.jsonl while "
            "still counting them in the summary. Use 256 for the first decoder "
            "experiment."
        ),
    )
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
        else cache_root / REPORT_ROOTS[args.trace_version] / "target_audit"
    )
    try:
        run_action_target_audit(
            ActionTargetAuditConfig(
                cache_root=cache_root,
                output_dir=output_dir,
                splits=tuple(
                    split.strip() for split in args.splits.split(",") if split.strip()
                ),
                trace_version=args.trace_version,
                max_operations=args.max_operations,
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
