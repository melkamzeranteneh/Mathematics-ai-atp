# Pointer baseline initialization comparison

This comparison isolates the effect of initializing pointer training from the
strongest tactic baseline. Both arms must use the same prepared dataset, split,
architecture, seed, optimizer, graph budgets, and epoch target. The only
changed field is `pointer.initialization_checkpoint`.

## Fixed inputs

- Config: `configs/pointer_gat_state_mean_attention_pretrained.json`
- Prepared data: the same normalized model-S-expression cache used by the
  source baseline
- Source baseline architecture: GATv2, hidden dimension 128, four layers,
  four heads, `state_mean_attention` readout
- Seed: 42
- Selection criterion: lowest validation combined loss

The initialization run fails before training if its architecture, edge mode,
node vocabulary, or tactic vocabulary differs from the source baseline.

## Random-initialization arm

```bash
python -m maths_ai.gnn_inference.scripts.run_training \
  --config maths_ai/gnn_inference/configs/pointer_gat_state_mean_attention_pretrained.json \
  --stages pointer \
  --prepared-root maths_ai/_support_files/artifacts/prepared/sexpr_full_model \
  --run-root maths_ai/gnn_inference/runs/pointer_init_comparison/random \
  --device cuda:0
```

## Baseline-initialization arm

Replace `BASELINE_BEST_PT` with the strongest compatible baseline checkpoint.

```bash
python -m maths_ai.gnn_inference.scripts.run_training \
  --config maths_ai/gnn_inference/configs/pointer_gat_state_mean_attention_pretrained.json \
  --stages pointer \
  --prepared-root maths_ai/_support_files/artifacts/prepared/sexpr_full_model \
  --run-root maths_ai/gnn_inference/runs/pointer_init_comparison/pretrained \
  --pointer.initialization_checkpoint BASELINE_BEST_PT \
  --device cuda:0
```

Do not compare runs that use different graph-budget policies or effective
dataset sizes. Pointer training is intentionally single-GPU because its
replica-dependent padded argument logits cannot be gathered safely by PyG
`DataParallel`.

## Required comparison record

Copy these fields from each run's `summary.json` and `eval_test.json`:

| Arm | Best epoch | Tactic top-1 | Tactic top-5 | Argument top-1 | Argument top-5 | Valid targets | Target coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | | | | | | | |
| Baseline initialized | | | | | | | |

For the initialized arm, also retain `pointer_initialization` from both
`config.json` and `summary.json`. It contains the checkpoint SHA-256, source
epoch, vocabulary hashes, transferred components, and the component left
randomly initialized. This makes the comparison auditable and prevents a
silently incompatible transfer.
