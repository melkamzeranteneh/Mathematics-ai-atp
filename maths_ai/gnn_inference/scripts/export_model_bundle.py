"""Export a training checkpoint as a publishable model bundle.

A checkpoint is not publishable as it stands.  It carries optimizer state that
nobody downstream needs, it is a pickle that executes code when loaded, its
config names directories on the training machine, and -- most importantly -- it
identifies its vocabularies only by a filesystem path, so the weights arrive
without the mapping that gives their numbers meaning.  This writes a bundle that
fixes all four: weights only, no pickle, redacted paths, and the vocabularies
shipped alongside with their hashes recorded.

Typical use, exporting the best pointer and the best baseline into one
repository that shares a single copy of the vocabularies::

    python -m maths_ai.gnn_inference.scripts.export_model_bundle \\
        --checkpoint runs/pointer_gat_.../run_20260812_095829/best.pt \\
        --output-dir artifacts/hf/Mathlib-Sexpr-GNN/pointer-gat \\
        --dataset jajostrains/Mathlib-Normalized-Sexpr

Pass ``--prepared-root`` for checkpoints written before the vocabularies were
embedded in them; pass ``--self-contained`` to put the vocabularies inside the
bundle directory instead of in a shared sibling ``vocab/``, for a bundle that
has to travel on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from maths_ai.gnn_inference.atp_lean_gnn.bundle import (
    HAS_SAFETENSORS,
    VALID_WEIGHTS_FORMATS,
    export_model_bundle,
    load_model_bundle,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a training checkpoint as a publishable model bundle."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint to export (best.pt or last.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Bundle directory to write. Created if it does not exist.",
    )
    parser.add_argument(
        "--prepared-root",
        type=str,
        default=None,
        help=(
            "Prepared dataset root to read the vocabularies from. Required only for "
            "checkpoints written before the vocabularies were embedded in them."
        ),
    )
    parser.add_argument(
        "--vocab-dir",
        type=str,
        default=None,
        help=(
            "Where to write the shared vocabularies. Defaults to a 'vocab' directory "
            "beside the bundle, so several bundles in one repository share one copy."
        ),
    )
    parser.add_argument(
        "--self-contained",
        action="store_true",
        help=(
            "Write the vocabularies inside the bundle directory instead of a shared "
            "sibling, for a bundle that must load after being moved on its own."
        ),
    )
    parser.add_argument(
        "--weights-format",
        type=str,
        default="auto",
        choices=list(VALID_WEIGHTS_FORMATS),
        help=(
            "Weight file format. 'auto' uses safetensors when installed and otherwise "
            "writes a pure tensor dict readable with weights_only=True."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset the model was trained on, e.g. 'org/Mathlib-Normalized-Sexpr'.",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=None,
        help="Dataset revision (commit sha or tag) the model was trained on.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Run directory to copy summary.json and metrics.jsonl from. Defaults to "
            "the checkpoint's own directory."
        ),
    )
    parser.add_argument(
        "--no-scorer",
        action="store_true",
        help=(
            "Skip the premise scorer even when the checkpoint carries one (a scorer "
            "checkpoint holds both the fine-tuned pointer and the scorer)."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "Skip reloading the bundle after writing it. Verification is on by default "
            "because a bundle that cannot be loaded is worse than no bundle."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    try:
        manifest = export_model_bundle(
            checkpoint_path=Path(args.checkpoint),
            output_dir=output_dir,
            prepared_root=None if args.prepared_root is None else Path(args.prepared_root),
            vocab_dir=None if args.vocab_dir is None else Path(args.vocab_dir),
            self_contained=bool(args.self_contained),
            weights_format=str(args.weights_format),
            dataset=args.dataset,
            dataset_revision=args.dataset_revision,
            run_dir=None if args.run_dir is None else Path(args.run_dir),
            include_scorer=not bool(args.no_scorer),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    paths = manifest.pop("_paths", {})
    print(f"Wrote {manifest['model_type']} bundle to {paths.get('bundle_dir', output_dir)}")
    print(f"  weights        {manifest['weights']} ({manifest['weights_format']})")
    print(f"  node vocab     {paths.get('node_vocab')}  ({manifest['num_node_labels']:,} labels)")
    print(f"  tactic vocab   {paths.get('tactic_vocab')}  ({manifest['num_tactics']:,} tactics)")
    print(f"  vocab source   {manifest['vocab_source']}")
    if "scorer" in manifest:
        scorer = manifest["scorer"]
        print(f"  scorer         {scorer['weights']} (mode={scorer['scoring_mode']})")
    if manifest["copied_run_files"]:
        print(f"  copied         {', '.join(manifest['copied_run_files'])}")
    if manifest["weights_format"] == "torch":
        if HAS_SAFETENSORS:
            print("  note           --weights-format=torch was requested explicitly.")
        else:
            print(
                "  note           safetensors is not installed, so weights were written "
                "as a pure tensor dict; it loads with weights_only=True and executes no "
                "pickle, but safetensors is preferred for publication."
            )

    if not args.no_verify:
        # Loading the bundle back is the only check that matters: it proves the
        # vocabularies resolve, their hashes match, and the weights fit the
        # architecture the config declares. Doing it here means a broken bundle
        # is caught by the person exporting it, not by whoever downloads it.
        try:
            loaded = load_model_bundle(output_dir, device="cpu")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"ERROR: bundle was written but does not load back: {exc}")
            return 1
        parameter_count = sum(p.numel() for p in loaded.model.parameters())
        print(
            f"  verified       reloaded as {loaded.model_type} "
            f"({parameter_count:,} parameters)"
        )

    print(json.dumps({"bundle": str(output_dir), "model_type": manifest["model_type"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
