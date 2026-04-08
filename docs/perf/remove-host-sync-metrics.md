# Remove Host Sync From Metric Accumulation

This branch removes avoidable host synchronization from training and validation metric accumulation relative to `miles30/performance`.

## What Changed

- Training epoch loss and training Dice totals stay on device during the batch loop.
- Validation loss, Dice totals, and processed-batch counts stay on device during the validation loop.
- Conversion to Python scalars now happens once per epoch or validation pass instead of once per batch.

## What Did Not Change

- Optimizer stepping is still once per batch.
- Loss math is unchanged.
- DistConv and DataLoader behavior are unchanged.
- CSV columns and printed summary fields are unchanged.

## Expected Effect

- Fewer per-batch host/device synchronization points.
- Lower CPU-side idle gaps around metric logging.
- Cleaner profiler timelines without changing convergence behavior.

## Evaluation

Throughput:

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
PROFILE_TORCH=ON torchrun-hpc -N 4 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 7 --dc-num-shards 1 2 2
```

## Acceptance Criteria

- `train_stats.csv` format is unchanged.
- No per-batch `.item()` remains in the train/validation accumulation path.
- Optimizer step cadence remains unchanged.
- Epoch `2+` duration improves or stays flat with cleaner CPU gaps in the trace.
