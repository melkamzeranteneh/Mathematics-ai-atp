# Exact S-expression extraction environment

The LeanDojo dataset was traced from Mathlib commit
`29dcec074de168ac2bf835a77ef68bbe069194c5` with Lean
`v4.10.0-rc1`. The extractor pins commit
`e6a8d53165a987d59c5780d2dd287d8ed4c95147` from the persistent
[`jajos12/Pantograph`](https://github.com/jajos12/Pantograph/tree/gnn-sexpr-v410)
fork. That commit contains both source-invocation tracing and the Lean-native
model S-expression serializer, plus Lean-elaborated tactic-term tracing; no
uncommitted patching is required.

Create the pinned environment once:

```bash
python -m maths_ai.gnn_inference.scripts.setup_sexpr_environment
```

The command prints the two paths needed by the extractor.  Then run, for
example:

```bash
python -m maths_ai.gnn_inference.scripts.generate_sexprs \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-repl maths_ai/_support_files/sexpr_environment/Pantograph/.lake/build/bin/repl \
  --splits train val test
```

The extractor compiles each original source file and matches dataset rows to
Pantograph's authentic tactic invocations using theorem name, before-state,
tactic, after-state, source file, and repository commit.  It never constructs
a replacement goal and never executes the dataset tactics.  Ambiguous matches
are failures rather than guessed cache entries.  Row caches are atomic and the
command is resumable.

After raw extraction, generate normalized sidecars without changing any raw
record:

```bash
python -m maths_ai.gnn_inference.scripts.generate_sexprs \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-repl maths_ai/_support_files/sexpr_environment/Pantograph/.lake/build/bin/repl \
  --splits train val test \
  --model-sexprs
```

Sidecars are stored under
`{prepared_root}/{split}/model_sexpr_v2/{row_index}.json`. Each contains the
SHA-256 digest of its corresponding raw record, so stale normalization is
rejected automatically while the costly raw extraction remains reusable.

Generate structured action traces without changing either existing cache:

```bash
python -m maths_ai.gnn_inference.scripts.generate_sexprs \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-repl maths_ai/_support_files/sexpr_environment/Pantograph/.lake/build/bin/repl \
  --splits train val test \
  --action-traces
```

These sidecars live at
`{prepared_root}/{split}/action_trace_v2/{row_index}.json`. They contain the
elaborated argument expression tree (including `:local`, `:global`, `:ctor`,
and `:app`), semantically confirmed fresh binder names, and the input goal's
stable local-context indices. Each sidecar is bound to its validated raw record
by SHA-256.

For a controlled pilot, first create one deterministic theorem-level
selection before S-expression extraction. It needs no raw cache. It stratifies
by tactic frequency, proof-state size, proof length, and context shape, while
using a shared randomized source-file priority so Pantograph compiles far fewer
Mathlib files:

```bash
python -m maths_ai.gnn_inference.scripts.build_sexpr_pilot \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output maths_ai/_support_files/artifacts/sexpr_pilot_30k.json \
  --train-rows 30000 \
  --val-rows 2000 \
  --test-rows 2000 \
  --seed 42
```

Pass that same file first to raw extraction, then normalized extraction, and
finally to both preprocessing runs with `--selection-manifest`. This ensures
the ablation uses identical theorem traces and row indices. Do not combine it
with first-N sampling (`--max-items` or `--sample-per-split`).

When benchmark validation/test extraction is intentionally deferred, a paired
representation ablation may instead use theorem-disjoint holdouts from the
validated train cache:

```bash
python -m maths_ai.gnn_inference.scripts.build_sexpr_pilot \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output maths_ai/_support_files/artifacts/sexpr_pilot_internal.json \
  --train-rows 30000 \
  --val-rows 2000 \
  --test-rows 2000 \
  --seed 42 \
  --require-cached-train \
  --evaluation-from-train
```

The `--train-rows` target is the total source pool before holdout, so the
logical training partition is approximately 26,000 rows. Such scores are
internal ablation measurements and must not be reported as official benchmark
validation/test accuracy.
