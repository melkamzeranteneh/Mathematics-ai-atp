#!/usr/bin/env python3
"""
Generate S-expressions for the LeanDojo benchmark dataset using Pantograph.

This script streams the dataset from HuggingFace, uses Pantograph to extract
S-expressions for each proof state, and saves them as JSON files alongside
the prepared data.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


async def generate_sexpressions(
    *,
    prepared_root: Path,
    splits: list[str] = ("train", "val", "test"),
    max_items_per_split: int | None = None,
    project_path: str = "maths_ai/lean_mathlib",
    resume: bool = True,
) -> dict[str, int]:
    """
    Generate S-expressions for the dataset using Pantograph.

    For each example in the dataset:
    1. Parse the proof state to extract goal and hypotheses
    2. Start a Pantograph goal with just the goal expression
    3. Apply intro tactics to introduce hypotheses
    4. Extract S-expressions for goal and hypotheses
    5. Save S-expressions as JSON alongside the prepared data
    """
    from maths_ai.gnn_inference.atp_lean_gnn.dataset import iter_dataset_rows, DATASET_NAME
    from maths_ai.gnn_inference.atp_lean_gnn.graph import (
        patch_pantograph_for_sexp,
        goal_state_to_proof_state,
    )
    from pantograph.server import Server

    # Patch Pantograph to return S-expressions
    patch_pantograph_for_sexp()

    print(f"Starting Pantograph server with project: {project_path}")
    server = await Server.create(
        project_path=project_path,
        imports=["Init", "Mathlib"],
        options={"printExprAST": True},
    )
    print("Pantograph server started")

    results = {}

    try:
        for split in splits:
            print(f"\n=== Processing split: {split} ===")

            sexpr_dir = Path("maths_ai/gnn_inference/artifacts/prepared/v1") / split / "sexpr"
            sexpr_dir.mkdir(parents=True, exist_ok=True)

            processed = 0
            skipped = 0
            failed = 0

            dataset_iter = iter_dataset_rows(
                dataset_name="cat-searcher/leandojo-benchmark-4-random",
                split=split,
                sample_limit=max_items_per_split,
            )

            for row in dataset_iter:
                idx = row.row_index
                sexpr_file = Path("maths_ai/gnn_inference/artifacts/prepared/v1") / split / "sexpr" / f"{idx:09d}.json"

                # Skip if already processed
                if resume and (Path("maths_ai/gnn_inference/artifacts/prepared/v1") / split / "sexpr" / f"{idx:09d}.json").exists():
                    skipped += 1
                    if skipped % 1000 == 0:
                        print(f"  [{split}] Skipped {skipped} (already exists)")
                    continue

                try:
                    state_str = row.state

                    # Parse the proof state: split into hypotheses and goal
                    # Format: "h1 : type1\nh2 : type2\n⊢ goal"
                    lines = state_str.split('\n')
                    hypotheses = []
                    goal = ""
                    for line in lines:
                        line = line.strip()
                        if line.startswith('⊢'):
                            goal = line[1:].strip()
                        elif ' : ' in line:
                            name, typ = line.split(':', 1)
                            hypotheses.append((name.strip(), typ.strip()))
                        elif ':' in line and ' : ' not in line:
                            name, typ = line.split(':', 1)
                            hypotheses.append((name.strip(), typ.strip()))

                    if not goal:
                        print(f"  [{split}] Row {row.row_index}: No goal found")
                        failed += 1
                        continue

                    # Start with just the goal
                    goal_state = await server.goal_start_async(goal)

                    # Introduce hypotheses
                    if hypotheses:
                        names = " ".join(name for name, _ in hypotheses)
                        goal_state = await server.goal_tactic_async(goal_state, f"intro {names}")

                    # Extract S-expressions
                    text_state, hyp_sexps, goal_sexp = goal_state_to_proof_state(goal_state)

                    # Save S-expressions
                    sexpr_data = {
                        "goal_sexp": goal_sexp,
                        "hyp_sexps": [{"name": name, "sexp": sexp} for name, sexp in hyp_sexps],
                        "text_state": text_state,
                    }

                    sexpr_dir = Path("maths_ai/gnn_inference/artifacts/prepared/v1") / split / "sexpr"
                    sexpr_dir.mkdir(parents=True, exist_ok=True)
                    sexpr_file = sexpr_dir / f"{row.row_index:09d}.json"
                    with open(sexpr_file, "w") as f:
                        json.dump(sexpr_data, f)

                    processed += 1
                    if processed % 100 == 0:
                        print(f"  [{split}] Processed {processed}...")

                except Exception as e:
                    print(f"  [{split}] Row {row.row_index}: FAILED - {e}")
                    failed += 1

            print(f"[{split}] Done: {processed} processed, {skipped} skipped, {failed} failed")
            results[split] = processed

    finally:
        server._close()
        print("Pantograph server closed")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate S-expressions for training dataset")
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("maths_ai/gnn_inference/artifacts/prepared/v1"),
        help="Path to prepared dataset root",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Max items per split (for testing)",
    )
    parser.add_argument(
        "--project-path",
        default="maths_ai/lean_mathlib",
        help="Path to Lean project for Pantograph",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already processed files",
    )
    args = parser.parse_args()

    print(f"Generating S-expressions for dataset")
    print(f"  Prepared root: {args.prepared_root}")
    print(f"  Splits: {args.splits}")
    print(f"  Max items per split: {args.max_items or 'unlimited'}")
    print(f"  Project path: {args.project_path}")
    print(f"  Resume: {not args.no_resume}")

    try:
        asyncio.run(main_async(
            prepared_root=args.prepared_root,
            splits=args.splits,
            max_items_per_split=args.max_items,
            project_path=args.project_path,
            resume=not args.no_resume,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


async def main_async(
    *,
    prepared_root: Path,
    splits: list[str] = ("train", "val", "test"),
    max_items_per_split: int | None = None,
    project_path: str = "maths_ai/lean_mathlib",
    resume: bool = True,
) -> dict[str, int]:
    from maths_ai.gnn_inference.atp_lean_gnn.dataset import iter_dataset_rows, DATASET_NAME
    from maths_ai.gnn_inference.atp_lean_gnn.graph import (
        patch_pantograph_for_sexp,
        goal_state_to_proof_state,
    )
    from pantograph.server import Server

    patch_pantograph_for_sexp()

    print(f"Starting Pantograph server with project: {project_path}")
    server = await Server.create(
        project_path=project_path,
        imports=["Init", "Mathlib"],
        options={"printExprAST": True},
    )
    print("Pantograph server started")

    results = {}

    try:
        for split in splits:
            print(f"\n=== Processing split: {split} ===")

            sexpr_dir = Path("maths_ai/gnn_inference/artifacts/prepared/v1") / split / "sexpr"
            sexpr_dir.mkdir(parents=True, exist_ok=True)

            processed = 0
            skipped = 0
            failed = 0

            dataset_iter = iter_dataset_rows(
                dataset_name="cat-searcher/leandojo-benchmark-4-random",
                split=split,
                sample_limit=max_items_per_split,
            )

            for row in dataset_iter:
                idx = row.row_index
                sexpr_file = Path("maths_ai/gnn_inference/artifacts/prepared/v1") / split / "sexpr" / f"{idx:09d}.json"

                if resume and (Path("maths_ai/gnn_inference/artifacts/prepared/v1") / split / "sexpr" / f"{idx:09d}.json").exists():
                    skipped += 1
                    if skipped % 1000 == 0:
                        print(f"  [{split}] Skipped {skipped} (already exists)")
                    continue

                try:
                    state_str = row.state

                    lines = state_str.split('\n')
                    hypotheses = []
                    goal = ""
                    for line in lines:
                        line = line.strip()
                        if line.startswith('⊢'):
                            goal = line[1:].strip()
                        elif ' : ' in line:
                            name, typ = line.split(':', 1)
                            hypotheses.append((name.strip(), typ.strip()))
                        elif ':' in line and ' : ' not in line:
                            name, typ = line.split(':', 1)
                            hypotheses.append((name.strip(), typ.strip()))

                    if not goal:
                        print(f"  [{split}] Row {row.row_index}: No goal found")
                        failed += 1
                        continue

                    # Start with just the goal
                    goal_state = await server.goal_start_async(goal)

                    # Introduce hypotheses
                    if hypotheses:
                        names = " ".join(name for name, _ in hypotheses)
                        goal_state = await server.goal_tactic_async(goal_state, f"intro {names}")

                    # Extract S-expressions
                    text_state, hyp_sexps, goal_sexp = goal_state_to_proof_state(goal_state)

                    # Save S-expressions
                    sexpr_data = {
                        "goal_sexp": goal_sexp,
                        "hyp_sexps": [{"name": name, "sexp": sexp} for name, sexp in hyp_sexps],
                        "text_state": text_state,
                    }

                    sexpr_dir = Path("maths_ai/gnn_inference/artifacts/prepared/v1") / split / "sexpr"
                    sexpr_dir.mkdir(parents=True, exist_ok=True)
                    sexpr_file = sexpr_dir / f"{row.row_index:09d}.json"
                    with open(sexpr_file, "w") as f:
                        json.dump(sexpr_data, f)

                    processed += 1
                    if processed % 100 == 0:
                        print(f"  [{split}] Processed {processed}...")

                except Exception as e:
                    print(f"  [{split}] Row {row.row_index}: FAILED - {e}")
                    failed += 1

            print(f"[{split}] Done: {processed} processed, {skipped} skipped, {failed} failed")
            results[split] = processed

    finally:
        server._close()
        print("Pantograph server closed")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate S-expressions for training dataset")
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("maths_ai/gnn_inference/artifacts/prepared/v1"),
        help="Path to prepared dataset root",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Max items per split (for testing)",
    )
    parser.add_argument(
        "--project-path",
        default="maths_ai/lean_mathlib",
        help="Path to Lean project for Pantograph",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already processed files",
    )
    args = parser.parse_args()

    print(f"Generating S-expressions for dataset")
    print(f"  Prepared root: {args.prepared_root}")
    print(f"  Splits: {args.splits}")
    print(f"  Max items per split: {args.max_items or 'unlimited'}")
    print(f"  Project path: {args.project_path}")
    print(f"  Resume: {not args.no_resume}")

    try:
        import asyncio
        asyncio.run(main_async(
            prepared_root=args.prepared_root,
            splits=args.splits,
            max_items_per_split=args.max_items,
            project_path=args.project_path,
            resume=not args.no_resume,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)