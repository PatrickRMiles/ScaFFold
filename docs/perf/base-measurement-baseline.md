# Performance Measurement Baseline

This branch establishes the shared evaluation harness for the isolated performance branches.

## What Changed

- Throughput runs no longer enable torch profiling by default.
- Torch profiling is now an explicit opt-in mode using `PROFILE_TORCH=ON`.
- Profiling captures a fixed short training window on global rank `0` only.
- Warmup is excluded from the profiled region.
- The standardized comparison matrix is:
  - `problem_scale=7` on `16` GPUs with `--dc-num-shards 1 2 2`
  - `problem_scale=8` on `32` GPUs with `--dc-num-shards 1 2 2`

## Profiling Behavior

When `PROFILE_TORCH=ON` is set:

- only global rank `0` is profiled
- cleanup and warmup run outside the profiler
- the training loop uses a fixed profiler schedule:
  - wait `5` batches
  - warmup `2` batches
  - record `8` batches

This keeps trace size and profiler overhead bounded while still capturing steady-state work.

## Evaluation Commands

Use these throughput runs for branch-to-branch comparisons:

```bash
torchrun-hpc -N 4 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 7 --dc-num-shards 1 2 2

torchrun-hpc -N 8 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 8 --dc-num-shards 1 2 2
```

Use this dedicated profiling mode only when a branch needs a trace:

```bash
PROFILE_TORCH=ON torchrun-hpc -N 4 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 7 --dc-num-shards 1 2 2
```

## Comparison Rules

- Compare `train_stats.csv` epoch durations from epoch `2` onward.
- Keep dataset, job shape, and sharding fixed across branches.
- Use profiler traces only as attribution support, not as the throughput baseline.
- Treat `problem_scale=9` as exploratory and outside the required comparison matrix.
