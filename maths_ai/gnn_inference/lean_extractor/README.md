# Exact S-expression extraction environment

The LeanDojo dataset was traced from Mathlib commit
`29dcec074de168ac2bf835a77ef68bbe069194c5` with Lean
`v4.10.0-rc1`. The extractor pins commit
`73781c2d58456e4bf369dadd4a501e1b78a0b177` from the persistent
[`jajos12/Pantograph`](https://github.com/jajos12/Pantograph/tree/gnn-sexpr-v410)
fork. That commit contains source-invocation tracing, the Lean-native model
S-expression serializer, Lean-elaborated tactic-term tracing, and compact
annotated tactic-syntax tracing; no uncommitted patching is required.

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

Version 2 traces are diagnostics rather than decoder targets. Because they are
built from Lean's fully elaborated terms, a local introduced inside the tactic
cannot be pointed at a goal hypothesis, and a nested `by ...` proof expands into
thousands of kernel nodes. The current decoder target is the compact original
tactic syntax annotated by Lean, written to a separate directory so the
elaborated traces stay available for comparison:

```bash
python -m maths_ai.gnn_inference.scripts.generate_sexprs \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-repl maths_ai/_support_files/sexpr_environment/Pantograph/.lake/build/bin/repl \
  --splits train val test \
  --source-syntax-traces
```

These sidecars live at
`{prepared_root}/{split}/action_trace_v3/{row_index}.json`. Each stores the
parsed syntax tree of the tactic as it was written, where every identifier leaf
carries the meaning Lean resolved for it: a stable local-context index, a
tactic-scoped binding, a global constant, or a constructor. Elaborated
expressions are deliberately not stored, only their byte ranges, so the
kernel-term blowup cannot return through the cache.

Before implementing or training a decoder, compile and audit the available
structured targets:

```bash
python -m maths_ai.gnn_inference.scripts.audit_action_targets \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --splits train \
  --trace-version v3 \
  --max-operations 256 \
  --force
```

The audit writes typed prefix-operation targets, coverage statistics, sequence
length percentiles, and representative unsupported cases beneath
`action_trace_extraction_v3/target_audit`; pass `--trace-version v2` to audit the
elaborated traces under `action_trace_extraction_v2/target_audit` instead. A
`LOCAL` operation is accepted only when its context index exists in that trace's
digest-validated local context. Targets longer than `--max-operations` are
excluded from `targets.jsonl` but still counted and sampled, so the cost of the
cap stays visible in the summary.

That audit measures the targets alone. To measure them against the graph the
pointer head actually selects from, and against the regex-and-arity supervision
they are meant to replace, run the argument coverage audit with
`--structured-traces`:

```bash
python -m maths_ai.gnn_inference.scripts.audit_argument_coverage \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --splits train \
  --structured-traces \
  --lemma-index maths_ai/_support_files/artifacts/lemma_index_v1 \
  --force
```

With `--structured-traces`, both metrics are computed over exactly the rows that
have a version-3 sidecar, so the comparison is not contaminated by rows only one
of them can see; rows outside that population are counted as
`rows_outside_trace_population`. The regex metric keeps its original denominator,
the argument slots the static tactic-arity table expects. The structured metric
uses a different denominator, the naming positions in the tactic Lean actually
parsed, so the two totals are not interchangeable and the report says so.

Each `LOCAL` position is resolved the way a decoder would have to resolve it:
the local-context index becomes the label `FV{context_index}`, that label is
looked up in the prepared node vocabulary, the matching node is found in the
graph, and its `premise_mask` entry decides between `local_selectable` and
`local_present_but_masked`. Failures are reported as distinct categories rather
than folded into one number: `local_absent_from_state` when the audited proof
state has no such context index, `local_name_mismatch` when the trace and the
graphed state disagree about which hypothesis an index names, and
`local_label_outside_vocab` or `local_node_absent` when the graph has no such
node. The name comparison ignores Lean's inaccessible-name marker and treats
anonymous names as unknown, so shadowed hypotheses do not raise false alarms.

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
