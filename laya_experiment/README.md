# Laya ATP Experiment

This folder converts the executable logic from `new_idea.ipynb` into a testable Python experiment. The goal is to compare:

- **System A:** the published GNN ranking over its top-10 tactic pool.
- **System B:** Laya reranking those same ten candidates, advancing its top five and selecting its first candidate.

The experiment measures tactic-ranking quality only. It does not execute tactics in Lean, so `recall@k` is not proof success.

## Current modules

- `config.py` — typed experiment settings and derived artifact/cache/output paths.
- `tactic_choose.py` — the 227 canonical tactic-family names used by the notebook vocabulary.
- `laya_adapter.py` — Lean-to-English translation, goal-first proof-state formatting, Laya choice-question construction, response validation, and position-balanced reranking.
- `laya_retrain.py` — text-only dataset translation/export and a trainer hook for a remote Laya installation.
- `evaluation.py` — recall, MRR, and GNN-versus-Laya transition metrics.
- `requirements.txt` — optional dependencies for the complete notebook conversion.

The remaining pipeline modules should use the existing ATP implementation under `maths_ai/gnn_inference/atp_lean_gnn` for graph construction, PyG conversion, model loading, and tactic normalization. Do not copy the notebook's `atp_core.py` unless a compatibility gap is demonstrated.

## Install dependencies

Use the repository virtual environment:

```bash
.venv/bin/pip install -r laya_experiment/requirements.txt
```

This requirements file intentionally does not install `torch` or `laya`.
Install a Torch build appropriate for the server first. Verify it before
continuing:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For a CPU-only server, the repository provides a constrained install file:

```bash
.venv/bin/pip install -r laya_experiment/requirements-torch-cpu.txt
```

For a GPU server, install the matching Torch wheel from the official PyTorch
index for that server's CUDA version instead. Do not install the CPU file on a
GPU machine if CUDA acceleration is required.

Then install Laya without dependency resolution so pip does not download a
second, potentially incompatible CUDA stack:

```bash
.venv/bin/pip install --no-deps -r laya_experiment/requirements-laya.txt
.venv/bin/python -c "import torch, laya; print(torch.__version__, torch.cuda.is_available(), 'laya import: ok')"
```

The core package dependencies are defined in the repository `pyproject.toml`.
If the server has no compatible Torch build, install the correct CPU or CUDA
wheel using the official PyTorch index before installing the files above. The
original failure happened because pip selected a new CUDA Torch wheel and ran
out of disk space while unpacking it.

## Lean translation

`laya_adapter.translate_lean_state()` keeps the goal first, translates common
logical operators such as `→`, `∧`, `∨`, `¬`, and `=` into English, and labels
local declarations as hypotheses. The original Lean state is not discarded;
retraining records store it in `raw_state` for auditability.

```bash
.venv/bin/python - <<'PY'
from laya_experiment import translate_lean_state, translate_tactic_step

print(translate_lean_state("h : x = y\n⊢ x = y"))
print(translate_tactic_step("rw", "h"))
PY
```

The translator is deliberately conservative. It improves the natural-language
context supplied to Laya, but it is not a Lean parser or proof checker.

## Text-only Laya retraining

`laya_retrain.py` prepares JSONL records from a dataset containing at least:

- `text_state` — the Lean proof state;
- `tactic` — the target tactic or tactic application;
- `candidates` — optional per-row candidate choices as a JSON list or comma-separated string.

If `candidates` is absent, provide an explicit, small candidate pool. Do not
send all 227 names to a `choice` head trained with a limited option budget.

Prepare records without running Laya:

```bash
.venv/bin/python -m laya_experiment.laya_retrain \
    data/train.jsonl outputs/laya_train.jsonl \
    --candidates rw,simp,exact,apply,assumption
```

The output contains the translated `state`, the original `raw_state`, a Laya
choice question, the normalized expected tactic, and a natural-language tactic
description. It never creates PyG graphs.

To connect a server-specific Laya trainer, expose a callable accepting
`records`, `output_path`, and optional keyword arguments, then pass it as an
import path:

```bash
.venv/bin/python -m laya_experiment.laya_retrain \
    data/train.parquet outputs/laya_train.jsonl \
    --candidate-column candidates \
    --trainer my_server_training:train_laya
```

The callable hook is intentional because Laya fine-tuning APIs can differ by
checkpoint/package version. The preparation and translation stage remains
stable, inspectable, and runnable before the remote training step.

## Verify the implemented core

From the repository root:

```bash
.venv/bin/python -m compileall -q laya_experiment
.venv/bin/python -c "from laya_experiment import ExperimentConfig; print(ExperimentConfig(n_rows=2))"
```

A minimal fake-agent check avoids downloading Laya:

```bash
.venv/bin/python - <<'PY'
from laya_experiment.laya_adapter import LayaAdapter

class FakeAgent:
    def predict(self, state, question):
        options = list(question["tactic"]["criteria"])
        probabilities = {name: 0.0 for name in options}
        probabilities[options[-1]] = 0.8
        probabilities[options[0]] = 0.2
        return {
            "answers": {
                "tactic": {
                    "probabilities": probabilities,
                    "choice": options[-1],
                    "answer_confidence": 0.8,
                }
            }
        }

adapter = LayaAdapter(FakeAgent(), "Which tactic fits this proof step?")
result = adapter.predict("h : P\\n⊢ P", ["exact", "assumption"])
assert result["order"] == ["assumption", "exact"]
assert set(result["probabilities"]) == {"exact", "assumption"}
print("fake Laya adapter check: passed")
PY
```

## Intended full run

The completed conversion should expose a CLI from the repository root:

```bash
.venv/bin/python -m laya_experiment --help
.venv/bin/python -m laya_experiment run --rows 20 --device cpu
```

A full run should perform these stages:

1. Download and verify the published GNN bundle and every parquet shard in the selected Hugging Face split.
2. Sample rows deterministically using the configured seed.
3. Build graphs from `text_state`, because the published node vocabulary matches the text-parser path.
4. Run the GNN and save its top-10 candidate pool.
5. Run one Laya choice request per row over exactly those ten candidates.
6. Optionally run position-balanced B1 and development-only B2 reranking.
7. Write resumable caches under `cache/` and reports under `outputs/`.

Expected report files include:

- `per_row_rankings.csv`
- `metrics.csv`
- `summary.md`
- `fig1_recall.png` through `fig4_rank_shift.png`

## Testing requirements for new pipeline modules

New modules should remain testable without network access, model downloads, or Laya installation. Use small fake artifacts and injected fake agents to test:

- configuration validation and path resolution;
- exact top-10 candidate restriction;
- goal-first state formatting;
- probability-based ordering and deterministic ties;
- cache round-tripping and resume behavior;
- per-row failure isolation;
- B1 rotation averaging;
- B2 held-out ranking behavior;
- metric and transition calculations;
- CSV, Markdown, and figure generation.

Before reporting benchmark numbers, run the real pipeline on the untouched test split and keep ranking evaluation separate from Lean execution evaluation.
