# Optimizer Step Overhead

This branch reduces optimizer-step overhead relative to `miles30/performance`.

## What Changed

- All training-path `optimizer.zero_grad(...)` calls now use `set_to_none=True`.
- Gradient clipping is now controlled by `gradient_clip_max_norm`.
- Clipping is disabled by default with `gradient_clip_max_norm: 0.0`.
- When clipping is disabled, the training loop skips both `grad_scaler.unscale_(optimizer)` and `clip_grad_norm_`.

## Public Interface

- New config key: `gradient_clip_max_norm`
- New CLI override: `--gradient-clip-max-norm`

Set `gradient_clip_max_norm: 1.0` to reproduce the previous clipping threshold.

## What Did Not Change

- Optimizer stepping is still once per batch.
- Loss math and Dice math are unchanged.
- Data loading and DistConv behavior are unchanged.

## Expected Effect

- Lower per-step optimizer overhead.
- Fewer full-parameter traversals in the benchmark default configuration.
- Cleaner separation between numerical stability tuning and throughput measurement.

## Evaluation

Default benchmark behavior now runs with clipping disabled:

```bash
torchrun-hpc -N 4 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 7 --dc-num-shards 1 2 2

torchrun-hpc -N 8 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 8 --dc-num-shards 1 2 2
```

To compare against the old clipping behavior:

```bash
torchrun-hpc -N 4 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 7 --dc-num-shards 1 2 2 \
  --gradient-clip-max-norm 1.0
```

## Acceptance Criteria

- Default config disables clipping.
- Enabling `--gradient-clip-max-norm 1.0` restores the old threshold.
- Throughput improves or stays flat in epoch `2+` timing.
- Training still runs correctly with and without clipping enabled.
