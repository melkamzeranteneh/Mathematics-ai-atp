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
  --prepared-root maths_ai/_support_files/artifacts/prepared/v2 \
  --sexpr-cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output-dir maths_ai/_support_files/artifacts/prepared/v2/reports/argument_coverage_v3 \
  --splits train \
  --structured-traces \
  --lemma-corpus maths_ai/_support_files/artifacts/lemmas/v1/corpus \
  --force
```

`--sexpr-cache-root` is needed whenever the graphs being audited were rebuilt
into a new prepared root while the S-expression and trace sidecars stayed in the
root they were extracted into; it defaults to `--prepared-root`. Only the splits
named by `--splits` must exist, so a partially rebuilt corpus can be audited
without placeholder manifests for the splits it does not yet contain.

Supply a candidate pool as well. Every `GLOBAL` and `CONSTRUCTOR` position is
classified `global_unchecked` without one, and that category is excluded from
`resolved_reference_coverage` rather than assumed resolvable, because "Lean gave
us a name" is not evidence that the name can be selected. With a pool the same
positions split into `global_library_lemma` and `global_outside_corpus`.

Which pool is used decides that split. `artifacts/lemmas/v1/corpus` was built
from the theorems the benchmark *proves* -- `extract_lemma_corpus_from_hf.py`
reads `row.theorem` and never looks at `row.tactic` -- but a tactic cites
lemmas, and the most frequently cited ones are the simplest. `mul_comm`,
`lt_of_le_of_lt` and `Set.inter_subset_left` are all proved in term mode, own no
traced tactic row, and therefore cannot appear in a target-derived corpus at any
row count. Measured over the 5187 train rows carrying a version-3 trace, that
corpus covered 2308 of 9528 lemma citations (24.22%), while the Mathlib
environment covers 9528 of 9528. Build the environment corpus instead:

```bash
python -m maths_ai.gnn_inference.scripts.extract_lemma_corpus_from_mathlib \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-repl maths_ai/_support_files/sexpr_environment/Pantograph/.lake/build/bin/repl \
  --output-dir maths_ai/_support_files/artifacts/lemmas/v2/corpus
```

`env.catalog` returns 318953 constants -- 219626 theorems, 88434 definitions, and
the constructors, inductives and recursors that tactics also cite -- each
prefixed with one character naming its kind. `env.inspect` supplies the type used
as the statement at about 2.7 ms per declaration. A full run took some 14 minutes
and wrote all 318953 records into 143 MB, against the 9.8 MB of the v1 corpus,
with no inspect failures and no unsafe declarations to skip. Auditing the same
5187 rows against it reports `global_library_lemma` 9528, no
`global_outside_corpus`, and resolved reference coverage of 96.1109% with 4043
rows (85.9481%) fully resolved -- against 63.84% and 1448 rows for the v1 pool.

Importing all of Mathlib overflows the default 8 MiB stack, which Lean reports as
`Stack overflow detected`; the script raises its own limit before spawning the
REPL, so no `ulimit -s` wrapper is needed. Because Mathlib imports Lean core and
Batteries transitively, no separate pass is required for those. Run one instance
at a time: two processes writing the same output directory leave a partially
written `lemmas.jsonl`, and the audit reads that as a small corpus rather than a
broken one, reporting a low coverage figure that looks like a result. Pass
`--names-only` to write just the `lemma_names.json` that `--lemma-index`
consumes: that finishes in seconds and is enough to label decoder targets, but
carries no statements and so cannot feed embeddings.

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

That resolution presupposes something the prepared corpus must actually
provide. The graph builder emits an `FV{context_index}` node only for
hypotheses that record a context index, and only the normalized
`model_sexpr_v2` sidecars record one; the raw source-faithful records carry
just a name and a type S-expression. A corpus preprocessed with
`--sexpr-variant raw` therefore contains no such node, and every local
reference is unresolvable regardless of any mask. Preprocessing defaults to
`--sexpr-variant model` for that reason, and the audit reports
`Context-index node labels in vocabulary` so a corpus built the other way is
diagnosed here rather than mistaken for a masking problem downstream:

```bash
python -m maths_ai.gnn_inference.atp_lean_gnn.preprocess \
  --output-root maths_ai/_support_files/artifacts/prepared/v2 \
  --use-sexpr \
  --sexpr-cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --sexpr-variant model \
  --selection-manifest maths_ai/_support_files/artifacts/sexpr_pilot_30k.json \
  --splits train val test
```

The hypothesis-name cross-check reads the same normalized sidecar. When a
record carries no context indices at all, the check is disabled rather than
reporting every index as absent from the state.

Measured on 5187 train rows with version-3 traces, that rebuild resolves
7572 of 7572 local references to an unmasked graph node with no name
disagreements, against 4355 argument slots the static arity table expects for
the same rows.

The model variant is the pointer representation the project now uses, and
`--sexpr-variant raw` is kept only for the representation ablation. The two are
not interchangeable supervision sources: once a hypothesis records a context
index, the graph builder makes the hypothesis *name* node an `sconst` and moves
selectability to the `FV` node. The regex-and-arity path loses the name node it
was accidentally pointing at, and its coverage on the same rows drops from
34.05% to 9.90%, with most of the difference reappearing as
`local_present_but_masked`. That drop is the expected consequence of the
representation change rather than a regression, and it is why a single corpus
cannot serve both schemes.

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
