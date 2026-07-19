import os, sys, json, time
from pathlib import Path

# Resolve paths relative to the project root (where this script lives)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # maths_ai/gnn_inference/scripts -> project root
os.environ.setdefault("CUDA_HOME", str(PROJECT_ROOT / "_work" / "cuda"))
os.environ.setdefault("GITHUB_ACCESS_TOKEN", "")

PROJECT = "_mathlib_src_clean"
SAMPLE = int(os.environ.get("SAMPLE", "50"))
TRACE_DIR = PROJECT_ROOT / "_work" / "traced"
OUT_DIR = PROJECT_ROOT / "_work" / "proofs"

def log(s):
    print(s, flush=True)

def main():
    from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo
    from lean_dojo_v2.lean_dojo.data_extraction.dataset import generate_benchmark, export_proofs

    t0 = time.time()
    log("[1] generate_benchmark (one-time Mathlib build)...")
    repo = LeanGitRepo.from_path(PROJECT)
    log(f"    repo={repo.name} commit={repo.commit} lean={repo.lean_version}")
    result = generate_benchmark(repo, str(TRACE_DIR), build_deps=True)
    log(f"[1] returned in {time.time()-t0:.0f}s")
    traced_repo = result[0]

    log("[2] enumerating theorems...")
    theorems = []
    for tf in traced_repo.traced_files:
        for attr in ("theorems", "traced_theorems"):
            if hasattr(tf, attr):
                vals = getattr(tf, attr)
                if isinstance(vals, (list, tuple)):
                    theorems.extend(vals); break
    log(f"    found {len(theorems)} theorems across {len(traced_repo.traced_files)} files")
    if not theorems:
        log("    NO theorems via file.theorems; TracedFile attrs: " + str([a for a in dir(traced_repo.traced_files[0]) if not a.startswith('_')]))
        return

    sample = theorems[:SAMPLE]
    splits = {"random": {"train": sample, "val": [], "test": []}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = export_proofs(splits, OUT_DIR, traced_repo)
    log(f"[3] exported {total} theorems")

    data = json.loads((OUT_DIR/"random"/"train.json").read_text())
    log(f"[4] train.json: {len(data)} theorems")
    for th in data[:5]:
        tacs = th.get("traced_tactics", [])
        log(f"    {th['full_name']}  ({len(tacs)} tactics)  first={tacs[0]['tactic'][:40] if tacs else None}")

if __name__ == "__main__":
    main()
