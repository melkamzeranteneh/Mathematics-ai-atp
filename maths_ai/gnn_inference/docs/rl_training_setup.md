# Starting RL training: weights, environment, and launch (remote server / JupyterLab)

Step-by-step operational guide: where the weight files go, what format the loaders
expect, how to prepare a fresh remote machine, and how to launch, monitor, resume, and
evaluate the RL run. The *mechanics* of what the loaders do internally are in
`docs/rl_process_walkthrough.md` and `docs/dev_plans/rl_training_driver.md`; this file is
the checklist.

## 0. The two hand-offs at a glance

```
pointer best.pt ──(hand-off A: filtered load + zero-residual copy)──▶ supervised actor-critic
                                                                        │ trains actor+critic
supervised actor-critic best.pt ──(hand-off B: strict full load)──▶ RL driver
                                                                        │ collect→train rounds
                                                              runs/rl_actor_critic/<stamp>/best.pt
```

The RL driver does **not** accept a pointer checkpoint directly — it strict-loads a full
`ActorCriticWithArgsClassifier` state dict. If you only have pointer weights, run
hand-off A first (even briefly) to produce one.

## 1. Checkpoint file format

Both loaders call `torch.load(path)` and read `checkpoint["model_state_dict"]`
(falling back to the whole object if that key is absent). So the expected file is:

- a **`.pt` file** written by `torch.save`,
- containing a dict `{"model_state_dict": <state_dict>, ...}` — extra keys like
  `optimizer_state_dict`/`epoch` are ignored — **or** a bare state dict.

This is exactly what this repo's training scripts write; checkpoints they produced need
no conversion. What the state dict must contain per hand-off:

| Hand-off | Expected tensors | Loaded by |
|---|---|---|
| A (pointer → supervised AC) | `TacticWithArgsClassifier`: `backbone.*`, `tactic_embedding.*`, `argument_selector.*` | `load_from_pointer_checkpoint` (filtered, shape-guarded, `strict=False`) |
| B (supervised AC → RL) | full `ActorCriticWithArgsClassifier` incl. `actor.*`, `critic.*` | RL driver (`strict=True`) |

Do not rename tensors from a foreign checkpoint to force a fit: matching shapes with a
different node/tactic vocab ordering loads without error and silently maps every
embedding row to the wrong token. The checkpoint is only valid together with the
`prepared_root` it was trained against.

## 2. Where to copy the files on the server

All paths are configuration, not convention — but use the repo layout so configs stay
readable. From the repo root:

```
runs/pointer_gnn/imported/best.pt            # ← your pointer weights (hand-off A input)
runs/actor_critic_gnn/<run>/best.pt          # ← produced by hand-off A (or copied in, if
                                             #    you already trained the AC phase elsewhere)
artifacts/prepared/v1/                       # ← the prepared dataset BOTH phases used:
    vocab/node_vocab.json                    #    these two files size & order every
    vocab/tactic_vocab.json                  #    embedding table — same files, all phases
    ...(split manifests / pyg tensors)...
```

Copy from your machine with `scp -r runs artifacts user@server:/path/to/new-maths/`, or
upload through the JupyterLab file browser (drag-and-drop; for multi-GB artifacts, zip
first and unzip in a terminal — the browser upload is per-file).

Note: the repo contains a symlink `maths_ai/gnn_inference/artifacts/prepared →
../../_support_files/artifacts/prepared` which is broken unless that target exists.
Either create the target directory or (simpler) point `prepared_root` in the configs at
the real absolute path of your prepared dataset.

## 3. Environment preparation (remote server, JupyterLab)

The RL phase runs live Lean and petta subprocesses, so the Python packages alone are not
enough. Work in a **JupyterLab terminal** (File → New → Terminal), not a notebook — the
driver is a long-running CLI process.

### 3.1 Python side

```bash
cd /path/to/new-maths
uv sync                          # installs torch, torch-geometric, pydantic, faiss-cpu,
                                 # and pantograph (from the PyPantograph git source)
uv add datasets                  # HuggingFace streaming — required by the driver's
                                 # dataset mode (iter_dataset_rows); not yet in pyproject
```

CUDA check (the prepared configs use `"device": "auto"`, which falls back to CPU —
fine for the RL phase, whose bottleneck is Lean, but verify what you got):

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 3.2 Lean / Pantograph

PyPantograph needs a Lean 4 toolchain (`elan`/`lake`) on PATH:

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh   # installs to ~/.elan
source ~/.elan/env
uv run python -c "
import asyncio
from pantograph.server import Server
async def main():
    s = await Server.create()
    st = await s.goal_start_async('forall (p : Prop), p -> p')
    print('pantograph OK:', st.goals[0].target)
asyncio.run(main())
"
```

The first `Server.create()` may compile Lean core — minutes, one-time.

### 3.3 petta (PLN)

The PLN reward path shells out to the `petta` binary. `PLNInference` finds it via, in
order: the `petta_bin` constructor arg, the `PETTA_BIN` environment variable, or
`shutil.which("petta")`. So either install it on PATH (e.g. `/usr/local/bin/petta`) or:

```bash
export PETTA_BIN=/path/to/petta          # put this in ~/.bashrc for JupyterLab terminals
```

Check:

```bash
uv run python -c "
from maths_ai.pln_inference.model import PLNInference
p = PLNInference()
r = p.evaluate('p -> p', hypotheses=['p : Prop'])
print('petta OK:', r.status, r.stv, 'fallback =', r.is_fallback)
"
```

`is_fallback=True` with status `petta_unavailable` means the binary wasn't found — the
run would still work but every Φ would be exploration noise instead of PLN.

### 3.4 Sanity: unit suite + live smoke

```bash
uv run python -m pytest maths_ai/gnn_inference/tests/ -q
# expected: all green except the pre-existing test_premise_pool failure
uv run python -m maths_ai.gnn_inference.scripts.rl_smoke
# expected: "[rl_smoke] OK — collect → harvest → one on-policy gradient step completed."
```

The smoke test needs no checkpoints (it builds a tiny fresh model) and validates the
whole live chain: Pantograph server, petta scoring, sampling search, harvest, gradient.

## 4. Hand-off A: pointer weights → supervised actor-critic checkpoint

Skip this section if you already have a trained `ActorCriticWithArgsClassifier`
checkpoint; go to section 5.

1. Edit `maths_ai/gnn_inference/configs/actor_critic_graphsage_state.json`:

```json
"prepared_root": "/abs/path/artifacts/prepared/v1",
"pretrained_pointer_checkpoint": "runs/pointer_gnn/imported/best.pt",
```

   The `model` block (`hidden_dim: 512, num_layers: 4, max_args: 3`) must equal the
   pointer run's architecture — the loader shape-checks every transferred tensor and
   raises a `ValueError` naming the mismatched keys rather than leaving the model at
   random init.

2. Run the supervised actor-critic phase:

```bash
uv run python maths_ai/gnn_inference/scripts/train_baseline.py \
  --model-type actor_critic \
  --config maths_ai/gnn_inference/configs/actor_critic_graphsage_state.json
```

3. Verify the warm start took, on the console: the
   `Warm-start: loaded N tensors ...` line, and **first-epoch tactic accuracy ≈ the
   pointer run's final accuracy** (the zero-residual property: the actor's step-0
   distribution is the supervised classifier's; near-random first-epoch accuracy means
   the warm start failed and the shape guard should have said why).

4. Output: `runs/actor_critic_gnn/<timestamp>/best.pt` — the input to hand-off B.

## 5. Hand-off B: launch the RL driver

1. Edit `maths_ai/gnn_inference/configs/rl_actor_critic.json`:

```json
"warmstart_checkpoint": "runs/actor_critic_gnn/<timestamp>/best.pt",
"prepared_root": "/abs/path/artifacts/prepared/v1",
"run_root": "runs/rl_actor_critic",
"device": "auto"
```

   Leave the architecture block (`hidden_dim` etc.) matching the checkpoint — the strict
   load makes any disagreement a startup error, which is the guard working, not a bug.
   The rest of the config (curriculum, BC anneal, budgets) ships with the plan's
   defaults; the two you are most likely to tune first are `theorems_per_round` (8) and
   `theorem_timeout_s` (120).

2. Launch **inside a persistent terminal** — a JupyterLab terminal dies with your browser
   session unless wrapped, so use tmux (or `nohup ... &`):

```bash
tmux new -s rl
cd /path/to/new-maths && source ~/.elan/env
uv run python maths_ai/gnn_inference/scripts/rl_train.py \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  2>&1 | tee rl_train.log
# detach: Ctrl-b d      reattach later: tmux attach -t rl
```

3. What startup prints, in order — each line is a checkpoint you can verify:
   `Warm start (strict): ...` → `Run dir: runs/rl_actor_critic/<stamp>` →
   `Theorem pool: N usable states, M dropped` → `Pool: X train / Y eval; curriculum
   window Z` → per-round lines:

```
Round 0: solved 2/8, trans 11, fail 19, return 0.213, loss 0.847, bc 0.500, 94.2s
```

## 6. Monitoring, resuming, evaluating

**Monitor** — from a JupyterLab notebook (this is the one place a notebook is the right
tool), plot the metrics file the driver appends per round:

```python
import json, pandas as pd
rows = [json.loads(l) for l in open("runs/rl_actor_critic/<stamp>/metrics.jsonl")]
train = pd.DataFrame([r for r in rows if "num_transitions" in r])
train.plot(x="round", y=["solved", "num_failures", "mean_return", "total_loss"], subplots=True)
```

Healthy early signs: `num_failures` trending down, `solved` and `mean_return` up,
`entropy` positive (not collapsing to 0), curriculum-widened lines appearing.

**Resume after a disconnect/preemption** — the driver checkpoints `last.pt` (model +
optimizer + RNG + round counter) every `checkpoint_every` rounds:

```bash
uv run python maths_ai/gnn_inference/scripts/rl_train.py \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --resume runs/rl_actor_critic/<stamp>
```

**Evaluate** — greedy proof rate on the same held-out pool, for the comparison that
matters (did RL beat its own warm start):

```bash
# baseline: the supervised warm start, before any RL
uv run python maths_ai/gnn_inference/scripts/rl_train.py \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --eval-only --checkpoint runs/actor_critic_gnn/<timestamp>/best.pt

# RL-tuned
uv run python maths_ai/gnn_inference/scripts/rl_train.py \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --eval-only --checkpoint runs/rl_actor_critic/<stamp>/best.pt
```

`best.pt` in the run dir is already model-selected by this metric (evaluated every
`eval_every` rounds), so the final artifact of the whole pipeline is
`runs/rl_actor_critic/<stamp>/best.pt`.

## 7. Failure modes at startup, and what each means

| Error | Cause | Fix |
|---|---|---|
| `Missing vocab file: .../vocab/node_vocab.json` | `prepared_root` wrong or symlink broken | point at the real prepared dataset directory |
| `RuntimeError: Error(s) in loading state_dict ... size mismatch` | architecture block ≠ checkpoint | set `hidden_dim`/`num_layers`/`max_args` to the checkpoint's values |
| `Missing key(s) in state_dict: "actor.base.weight" ...` | a pointer checkpoint was given to hand-off B | run hand-off A first |
| `Warm-start shape mismatch (hidden_dim disagreement...)` | hand-off A config ≠ pointer architecture | same fix as above, in the AC config |
| every PLN result `is_fallback=True` | petta not found | install petta / set `PETTA_BIN` |
| `Server.create()` hangs or errors | no Lean toolchain | install elan, `source ~/.elan/env` |
| `ModuleNotFoundError: datasets` | streaming dep not installed | `uv add datasets` |
