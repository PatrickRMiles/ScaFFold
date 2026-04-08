# Dice Path Rewrite

This branch rewrites the Dice path relative to `miles30/performance` while leaving cross-entropy semantics unchanged.

## What Changed

- Removed dense one-hot expansion of the full 3D label volume from train, warmup, and validation.
- Replaced the Dice statistic construction with classwise scatter-based accumulation from integer labels.
- Switched the spatial reduction from one all-reduce per mesh dimension to one all-reduce on a flattened spatial group.

## What Did Not Change

- Cross-entropy loss semantics are unchanged.
- Reported Dice still excludes background class `0`.
- Optimizer stepping, DataLoader behavior, and user-visible sharding arguments are unchanged.

## Expected Effect

- Less temporary memory traffic in the Dice path.
- Lower communication overhead for Dice and sharded CE reductions.
- Better GPU saturation at larger spatial scales where one-hot tensors are expensive.

## Evaluation

```bash
torchrun-hpc -N 4 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 7 --dc-num-shards 1 2 2

torchrun-hpc -N 8 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 8 --dc-num-shards 1 2 2
```

Optional trace:

```bash
PROFILE_TORCH=ON torchrun-hpc -N 8 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 8 --dc-num-shards 1 2 2
```

## Acceptance Criteria

- No dense label one-hot allocation remains in train, warmup, or eval.
- Reported Dice stays finite and continues to exclude background.
- CE loss semantics remain unchanged.
- Epoch `2+` duration improves or stays flat for the required `ps7`/`ps8` comparison matrix.
