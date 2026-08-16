# Porting the Pantograph fixes onto MCTS-AC-branch

## Context

`MCTS-AC-branch` branched from `AC-branch` at `5ffd525c`, before either of the two commits
that fixed the Pantograph integration:

| Commit | Content |
|---|---|
| `b5ba7e70` | first crash-recovery attempt: `_restart_server`, `server_kwargs`, elaboration guard |
| `1476ca6c` | the seven RL-search-log defects; supersedes and corrects `b5ba7e70` |

Both branches reach Lean through the same code — `HybridReasoner._start_state` calling
`Server.goal_start_async` — so this branch carries the same defects. `Server.create()` with
no `project_path` leaves `lean_path` at `None`, the REPL subprocess runs with no `LEAN_PATH`
and sees only core `Init`, and every goal mentioning Mathlib types or notation fails to
elaborate: the run collects zero transitions in every round. A REPL that panics is never
reaped, so the first crash silently kills the remainder of the run.

This branch also carries features `AC-branch` does not have: PUCT/HTPS repeated simulation,
per-edge visit statistics (`N` visits, `W` accumulated backup value), soft critic targets,
and minimal-hypertree mining. Two of those consume `NodeStatus.DEAD` as a training label,
which makes one AC-branch fix actively wrong here if ported verbatim — see
[The elaboration label](#the-elaboration-label-the-one-genuinely-new-design-decision).

Intended outcome: this branch runs against the compiled `maths_ai/lean_mathlib` project in
both search modes, survives REPL crashes, and never feeds a fabricated 0.0 into the critic
or the visit statistics.

## What is ported, and from where

Port the **post-`1476ca6c` state**, not the two commits in sequence. `b5ba7e70`'s
`_restart_server` calls `self.server.close()`, which does not exist on `Server`; the
resulting `AttributeError` was swallowed by a bare `except` and the dead subprocess leaked.
`1476ca6c` replaced it with `_close()`. Replaying history would add the defect and remove it
again.

Of the five changes in `pantograph_server_restart_fix.md`, two survive in their final form:
Change 1 (guard the `_start_state` call site) and Change 4 (restart inside `_start_state`).
Changes 2, 3 and 5 — a `server_kwargs` dict threaded from the driver — were replaced by
`PantographEnv`, a frozen value object naming one Lean environment: `source_root` (the Lake
project whose compiled `.olean` artifacts the REPL sees), `pantograph_repl` (the binary to
exec), `imports`, `options`, and `timeout`.

### Copy verbatim

`git diff --stat HEAD 1476ca6c^` reports no difference for these three, so
`git checkout 1476ca6c -- <path>` reproduces the fixed file exactly:

- `maths_ai/gnn_inference/atp_lean_gnn/state.py` — `parse_state` drops `^case\s+\S+$` lines
- `maths_ai/gnn_inference/scripts/rl_smoke.py` — gains `--source-root`, `--pantograph-repl`
- `maths_ai/gnn_inference/tests/test_graph_pipeline.py` — three case-label tests

Plus three files absent here entirely: `maths_ai/hybrid_reasoner/pantograph_env.py`,
`maths_ai/gnn_inference/tests/test_pantograph_env.py`,
`maths_ai/gnn_inference/tests/test_tactic_rendering.py`.

The copied `rl_smoke.py` builds the reasoner without the MCTS keywords, so it stays on the
legacy best-first path. That is deliberate: it probes the Lean environment, not the search
mode.

### Hand-port

| File | Divergence vs AC pre-fix |
|---|---|
| `maths_ai/hybrid_reasoner/joint_inference.py` | 420 lines |
| `maths_ai/gnn_inference/atp_lean_gnn/rl_training_driver.py` | 197 |
| `maths_ai/gnn_inference/tests/test_rl_training_driver.py` | 133 |
| `maths_ai/gnn_inference/configs/rl_actor_critic.json` | 20 |

The divergence is structural, not incidental: both branches implemented the PLN kill switch
independently, and this branch added the whole MCTS layer on top. A cherry-pick conflicts
across most hunks of the first two files, and a mis-resolved hunk drops a fix without
failing any test.

## The elaboration label: the one genuinely new design decision

`pantograph_server_restart_fix.md` Change 1 marks a node exhausted when its goal cannot be
elaborated:

```python
except (ServerError, ParseError) as exc:
    graph.mark_node_exhausted(node.id, note=f"elaboration error: {exc}")
```

`ProofHypergraph._recompute_node` (`maths_ai/hybrid_reasoner/hypergraph.py:465-470`) turns a
node that is `exhausted` with zero edges into `NodeStatus.DEAD`. On `AC-branch` DEAD had one
consumer and no consequence beyond ending that branch. On this branch DEAD is a **training
label** in three places:

1. `extract_critic_samples` (`search_harvest.py:208-216`) emits `target = 0.0`
2. `backup_values` (`search_harvest.py:139-141`) memoizes `0.0`, which flows through
   `_and_combine` into every harvested transition's `value_target`, `children_value`, and
   `return_` (`search_harvest.py:339, 347-359`)
3. `_backup_simulation`'s `node_value` (`joint_inference.py:601-603`) returns `0.0`, which is
   multiplied into the traversed edge's `W` — so the PUCT prior against that whole subtree
   drops for the rest of the search

A failed elaboration is not evidence that the goal is unprovable. It is a defect in goal
reconstruction or in the Lean environment — the case-label defect fixed in `state.py`
produced exactly this, and so does a `source_root` pointing at an unbuilt project. Labelling
it 0.0 teaches the critic that a provable goal is hopeless and punishes the parent tactic
that produced it.

**Decision: an explicit `unelaborated` flag, excluded from every value path.**

- `ProofNode` (`hypergraph.py:59-74`) gains `unelaborated: bool = False`, set beside the
  existing `exhausted`/`note` fields and reported in `ProofNode.summary()`.
- `ProofHypergraph.mark_node_exhausted` (`hypergraph.py:326-335`) gains a keyword
  `unelaborated: bool = False`, so the one existing shared entry point stays the only way to
  close a node. `_recompute_node` is untouched: the node still becomes DEAD, so search
  termination, frontier ranking, and `EdgeStatus` propagation behave exactly as now.
- `extract_critic_samples` skips nodes with `unelaborated` set — no sample, rather than a
  guessed one.
- `backup_values` treats such a node as unresolved: `_soft_target` when its edges carry
  enough visits, else `cfg.unresolved_leaf_value`. This is the adjacent shared component
  carrying the same mismatch, so it is fixed in the same change rather than left for the
  next caller to discover.
- `_backup_simulation`'s `node_value` uses `self._leaf_value(node)` for such a node — the
  critic's own estimate, which is what an unexpanded leaf already gets.

Alternative considered and rejected: leave the AC behaviour and let the `max_dead_rounds`
halt catch the misconfiguration. It does catch a wholly broken environment, but not the
common case — a handful of unparseable goals per round quietly poisoning the critic while
the run looks healthy.

Alternative considered and rejected: skip `mark_node_exhausted` entirely and leave the node
OPEN. The best-first loop would re-select the same node forever, and
`_select_partial_hypertree` would hand it back as a leaf on every simulation.

## Implementation steps

**1. Copy the five verbatim files** (`git checkout 1476ca6c -- <path>` for the three that
exist there, which also creates the three new ones).

**2. `hypergraph.py`** — add `ProofNode.unelaborated`, the `mark_node_exhausted` keyword, and
the `summary()` entry.

**3. `joint_inference.py`** — port, in this order:

- imports: `ServerError`, `ParseError` from `pantograph.server`; `perf_counter`;
  `PantographEnv`
- module level: `_LEAN_IDENT_RE`, `_BRACKET_REQUIRED_TACTICS`, `_BRACKET_OPTIONAL_TACTICS`,
  `_server_is_dead`, `render_tactic_command`
- `PantographExecutor.apply`: replace the two-line argument join with `render_tactic_command`
  and the named error for an unrenderable candidate
- `HybridReasoner.__init__`: add `env: PantographEnv | None = None`, store
  `self._env = env or PantographEnv()`. This branch has no `server_kwargs` parameter to
  remove — the MCTS keywords (`selection_policy`, `num_simulations`, `sim_batch_size`,
  `puct_c`) stay exactly as they are
- `_restart_server`: new method (`_close()`, `self._env.create_server()`, reinstall on
  `self.executor`)
- `_start_state`: split into `_start_state` (the restart wrapper) and `_goal_state_for` (the
  `goal_start_async` + `intro` pair). This is the single seam both search modes share, so no
  MCTS-specific restart logic is needed
- `main()`: build a `PantographEnv` instead of the inline `server_kwargs` dict
  (`joint_inference.py:813-824`), call `env.verify()`, then `env.create_server()`

**4. The elaboration guard, at one call site, not two.** AC-branch put it in `_expand`. On
this branch `_expand` and `_expand_leaves` both delegate to `_execute_and_link`
(`joint_inference.py:704`), which is where `_start_state` is called
(`joint_inference.py:713`). Guarding there covers best-first and MCTS with one copy:

```python
try:
    state = await self._start_state(node.goal)
except (ServerError, ParseError) as exc:
    console_print(f"  [Node {node.id} SKIP] goal elaboration failed: {exc}")
    graph.mark_node_exhausted(node.id, note=f"elaboration error: {exc}", unelaborated=True)
    self._on_expansion_complete(node)
    return
```

The `_on_expansion_complete(node)` call is required and is not in the AC version, which had
no such hook. Without it `RLHybridReasoner._flush_pending` never runs for this node, so the
sampled actions for a batched proposal stay in `self._pending` until `prove()`'s final sweep
and are then attributed as failure records — after the node they belong to is gone.

**5. `search_harvest.py`** — the two `unelaborated` exclusions described above, and a note in
the module docstring's value-backup list, which currently states DEAD → 0.0 without
qualification.

**6. `rl_training_driver.py`** — port each piece into the current structure:

- `RLTrainingConfig`: add `source_root`, `pantograph_repl`, `pantograph_imports`,
  `server_timeout_s`, `max_dead_rounds`; extend `_PATH_FIELDS` with `source_root` and
  `pantograph_repl`. `__post_init__`'s `resolve_search_params` call is untouched
- `pantograph_env(cfg)`: the resolver both server sites call, so the initial server and a
  post-crash restart are described by one value
- `_METAVARIABLE_RE` / `_has_metavariable`, and the two rejection points in
  `build_theorem_pool` (dataset rows and file rows), counted into the existing `dropped`
- `bc_weight_at_round`: unchanged code, corrected docstring — it is the *caller* that changes
- `save_checkpoint`: add `anneal_rounds_done: int = 0` to the existing keyword-only block
  (`optimizer_htps`, `tactic_queue`, `critic_queue`) and into `payload`. All four existing
  call sites pass it
- the resume block: restore `anneal_rounds_done` with the same `state.get(..., 0)` pattern
  already used for the HTPS queues
- the live-reasoner site (`rl_training_driver.py:566-576`): `env = pantograph_env(cfg)`,
  `env.verify()`, `server = await env.create_server()`, pass `env=env`. The MCTS keywords
  already in that call stay
- the round loop: `dead_rounds` counter and the `max_dead_rounds` halt; `bc_weight_at_round(
  anneal_rounds_done, cfg)`; `anneal_rounds_done += 1` inside the `if results:` branch,
  which is also where the single on-policy optimizer step lives, so the counter and the step
  cannot diverge; `anneal_rounds_done` in the metrics row; `rej`/`err` on the console line
- `driver_main`: the four new flags and the `--eval-only` path's `env.verify()` before the
  checkpoint load. The `--eval-only` reasoner also gains `env=env`

The `max_dead_rounds` halt is placed after `collect_round` and before the train step, so a
round that collected nothing raises before `anneal_rounds_done` moves.

**7. `configs/rl_actor_critic.json`** — add the five keys with the same null/default values
AC uses, alongside the existing MCTS keys. An all-null environment reproduces a bare
`Server.create()`, so behaviour is unchanged until `source_root` is set.

**8. Tests** — port AC's `test_rl_training_driver.py` additions (BC-anneal-by-step,
`max_dead_rounds`, metavariable rejection, `pantograph_env` resolution) into the existing
class layout, which already has `ConfigTests`, `PoolTests`, `BCAnnealTests`, `RoundLoopTests`,
`EvalTests`, `HTPSDriverTests`, `PLNKillSwitchDriverTests`. The existing `_RaisingReasoner`
fake already simulates a search that raises, which is what a dead round needs. Add to
`test_search_harvest.py`: an unelaborated node yields no critic sample and does not back up
0.0 through its parent.

## Verification

```bash
# 1. Full suite. Expect test_premise_pool.py::test_builds_unified_pool to fail — it fails
#    identically with no changes applied and touches none of this code.
pytest maths_ai/gnn_inference/tests/

# 2. Environment probe, no model and no training loop. Without --source-root the goals
#    fail to elaborate; with it they must not.
python -m maths_ai.gnn_inference.scripts.rl_smoke \
    --source-root maths_ai/lean_mathlib

# 3. Legacy best-first, live Lean, three rounds.
python -m maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver \
    --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
    --source-root maths_ai/lean_mathlib

# 4. Same config with selection_policy="puct" and num_simulations set, so the MCTS path
#    runs through the same restart seam and the same guard.
```

Round lines should read `trans N` with N > 0 and `err 0`. `rej` counts tactics the executor
refused inside searches that ran; `err` counts whole searches that raised or timed out.
`bc` must hold its value across a round that collected nothing.

Both `maths_ai/lean_mathlib/lean-toolchain` and the bundled PyPantograph REPL are
`leanprover/lean4:v4.29.1` on this machine, which `env.verify()` checks before any model is
constructed — `.olean` artifacts carry the compiler version that wrote them, and a mismatch
surfaces as `KeyError('fragment')` several frames into goal parsing.

Do not run `lake` setup, checkout, restore, or build against `maths_ai/lean_mathlib` while an
extraction process is using it; per `RL_DRIVER_PANTOGRAPH_INTEGRATION.md` the shared
directories are read-only to the RL driver.

## Alternatives considered for the port method

| Approach | Verdict |
|---|---|
| `git cherry-pick 1476ca6c` | Conflicts across most hunks of `joint_inference.py` and `rl_training_driver.py`, because the PLN kill switch was implemented independently on each branch. A mis-resolved hunk drops a fix with no failing test. |
| `git merge AC-branch` | Same conflicts, plus it drags in every unrelated AC commit since `5ffd525c`. |
| Cherry-pick `b5ba7e70` then `1476ca6c` | Adds `self.server.close()`, which does not exist on `Server`, then removes it. Two conflict resolutions instead of one. |
| **Copy the five unchanged files, hand-port the four diverged ones (chosen)** | The mechanical part stays mechanical and byte-exact; review attention goes to the four files where the two branches actually differ. |

