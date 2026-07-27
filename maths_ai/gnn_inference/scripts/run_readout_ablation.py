#!/usr/bin/env python3
"""Run controlled GAT readout ablations across independent GPUs."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "maths_ai" / "gnn_inference" / "configs"
DEFAULT_PREPARED_ROOT = (
    REPO_ROOT / "maths_ai" / "_support_files" / "artifacts" / "prepared" / "v1"
)
VARIANT_CONFIGS = {
    "state_mean_attention": CONFIG_ROOT / "baseline_gat_state_mean_attention.json",
    "state_max_attention": CONFIG_ROOT / "baseline_gat_state_max_attention.json",
    "state_mean_max_attention": (
        CONFIG_ROOT / "baseline_gat_state_mean_max_attention.json"
    ),
}


def _parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=DEFAULT_PREPARED_ROOT,
        help="Prepared dataset shared by every ablation run.",
    )
    parser.add_argument(
        "--gpus",
        type=_parse_csv,
        default=["0", "1"],
        help="Comma-separated CUDA indices; one training process runs per GPU.",
    )
    parser.add_argument(
        "--variants",
        type=_parse_csv,
        default=list(VARIANT_CONFIGS),
        help=f"Comma-separated subset of: {', '.join(VARIANT_CONFIGS)}.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO_ROOT
        / "maths_ai"
        / "gnn_inference"
        / "runs"
        / "readout_ablation_logs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scheduled commands without starting training.",
    )
    return parser


def _command(variant: str, gpu: str, prepared_root: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "maths_ai.gnn_inference.scripts.run_training",
        "--config",
        str(VARIANT_CONFIGS[variant]),
        "--stages",
        "baseline",
        "--prepared-root",
        str(prepared_root),
        "--device",
        f"cuda:{gpu}",
        "--experiment-name",
        f"readout_ablation_{variant}",
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = sorted(set(args.variants) - set(VARIANT_CONFIGS))
    if unknown:
        raise SystemExit(
            f"Unknown variants: {', '.join(unknown)}. "
            f"Choose from: {', '.join(VARIANT_CONFIGS)}."
        )
    if len(set(args.variants)) != len(args.variants):
        raise SystemExit("Each readout variant may be scheduled only once.")
    if len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("GPU indices must be unique.")

    prepared_root = args.prepared_root.resolve()
    if not prepared_root.is_dir():
        raise SystemExit(f"Prepared dataset does not exist: {prepared_root}")

    scheduled_commands = [
        (
            variant,
            args.gpus[index % len(args.gpus)],
            _command(
                variant,
                args.gpus[index % len(args.gpus)],
                prepared_root,
            ),
        )
        for index, variant in enumerate(args.variants)
    ]
    if args.dry_run:
        for variant, gpu, command in scheduled_commands:
            print(f"[{variant}] gpu={gpu}: {' '.join(command)}")
        return 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_root.resolve() / timestamp
    log_dir.mkdir(parents=True, exist_ok=False)
    pending = list(args.variants)
    available_gpus = list(args.gpus)
    active: dict[str, tuple[str, subprocess.Popen[bytes], object, float]] = {}
    results: list[dict[str, object]] = []

    try:
        while pending or active:
            while pending and available_gpus:
                gpu = available_gpus.pop(0)
                variant = pending.pop(0)
                command = _command(variant, gpu, prepared_root)
                log_path = log_dir / f"{variant}.log"
                log_handle = log_path.open("wb")
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                active[variant] = (
                    gpu,
                    process,
                    log_handle,
                    time.monotonic(),
                )
                print(
                    f"Started {variant} on cuda:{gpu} (pid={process.pid}); "
                    f"log={log_path}",
                    flush=True,
                )

            time.sleep(1)
            for variant, (gpu, process, log_handle, started) in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log_handle.close()
                elapsed = time.monotonic() - started
                results.append(
                    {
                        "variant": variant,
                        "gpu": gpu,
                        "return_code": return_code,
                        "elapsed_seconds": elapsed,
                        "log": str(log_dir / f"{variant}.log"),
                    }
                )
                del active[variant]
                available_gpus.append(gpu)
                print(
                    f"Finished {variant} on cuda:{gpu}: exit={return_code}, "
                    f"elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Interrupting active ablation processes...", flush=True)
        for _variant, (_gpu, process, _log_handle, _started) in active.items():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for _variant, (_gpu, process, log_handle, _started) in active.items():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            log_handle.close()
        return 130

    summary_path = log_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "prepared_root": str(prepared_root),
                "gpus": args.gpus,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Ablation summary: {summary_path}")
    return 0 if all(item["return_code"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
