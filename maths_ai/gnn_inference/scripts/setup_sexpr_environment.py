#!/usr/bin/env python3
"""Build the exact Mathlib/Pantograph environment used by S-expression extraction."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


MATHLIB_URL = "https://github.com/leanprover-community/mathlib4.git"
MATHLIB_COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"
PANTOGRAPH_URL = "https://github.com/leanprover/Pantograph.git"
PANTOGRAPH_COMMIT = "22ddfaaf2124d323dec59220f567273f01623458"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.10.0-rc1"


def _run(*command: str, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _output(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checkout(
    url: str, commit: str, destination: Path, *, allow_existing_patch: bool = False
) -> None:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", "--filter=blob:none", "--no-checkout", url, str(destination))
    if not (destination / ".git").exists():
        raise RuntimeError(f"Refusing to reuse non-git directory: {destination}")
    dirty = _output("git", "status", "--porcelain", cwd=destination)
    if dirty and not (
        allow_existing_patch
        and _output("git", "rev-parse", "HEAD", cwd=destination) == commit
    ):
        raise RuntimeError(f"Checkout has local modifications; refusing to overwrite: {destination}")
    if dirty:
        return
    _run("git", "fetch", "--depth=1", "origin", commit, cwd=destination)
    _run("git", "checkout", "--detach", commit, cwd=destination)


def _apply_patch(repository: Path, patch: Path) -> None:
    forward = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=repository,
        capture_output=True,
    ).returncode
    if forward == 0:
        _run("git", "apply", str(patch), cwd=repository)
        return
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=repository,
        capture_output=True,
    ).returncode
    if reverse != 0:
        raise RuntimeError("Pantograph patch is neither applicable nor already applied.")
    print("Pantograph patch is already applied.")


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "maths_ai" / "_support_files" / "sexpr_environment",
    )
    parser.add_argument("--skip-mathlib-cache", action="store_true")
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    mathlib = output_root / "mathlib4"
    pantograph = output_root / "Pantograph"
    patch = repo_root / "maths_ai" / "gnn_inference" / "lean_extractor" / "pantograph_v410_invocation_sexprs.patch"

    _run("elan", "toolchain", "install", LEAN_TOOLCHAIN)
    _checkout(MATHLIB_URL, MATHLIB_COMMIT, mathlib)
    if not args.skip_mathlib_cache:
        _run("lake", "exe", "cache", "get", cwd=mathlib)
    _checkout(PANTOGRAPH_URL, PANTOGRAPH_COMMIT, pantograph, allow_existing_patch=True)
    _apply_patch(pantograph, patch)
    _run("lake", "build", cwd=pantograph)

    repl = pantograph / ".lake" / "build" / "bin" / "repl"
    if not repl.is_file():
        raise RuntimeError(f"Pantograph build completed without REPL binary: {repl}")
    print("\nS-expression environment is ready:")
    print(f"  --source-root {mathlib}")
    print(f"  --pantograph-repl {repl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
