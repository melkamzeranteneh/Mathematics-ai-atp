# Exact S-expression extraction environment

The LeanDojo dataset was traced from Mathlib commit
`29dcec074de168ac2bf835a77ef68bbe069194c5` with Lean
`v4.10.0-rc1`. The extractor also pins Pantograph commit
`22ddfaaf2124d323dec59220f567273f01623458`. A newer checkout is not
replay-compatible.

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
