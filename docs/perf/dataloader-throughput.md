# DataLoader Throughput

This branch improves the input pipeline relative to `miles30/performance`.

## What Changed

- Added configurable DataLoader worker count and prefetch factor.
- Enabled persistent workers automatically when worker count is greater than zero.
- Cached image and mask paths during dataset construction instead of doing per-sample `glob` lookups.
- Added a fast integer lookup-table path for mask remapping when mask values are compatible with direct indexing.

## Public Interface

- New config key: `dataloader_num_workers`
- New config key: `dataloader_prefetch_factor`
- New CLI override: `--dataloader-num-workers`
- New CLI override: `--dataloader-prefetch-factor`

Branch defaults:

- `dataloader_num_workers: 4`
- `dataloader_prefetch_factor: 2`

## What Did Not Change

- Dataset file format is unchanged.
- Tensor shapes, dtypes, and channel ordering are unchanged.
- Loss math, optimizer behavior, and DistConv behavior are unchanged.

## Expected Effect

- Lower filesystem metadata overhead in `__getitem__`.
- Better overlap between file I/O, CPU preprocessing, and GPU work.
- Reduced label-remap cost for the common integer-mask case.

## Evaluation

```bash
torchrun-hpc -N 4 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 7 --dc-num-shards 1 2 2

torchrun-hpc -N 8 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 8 --dc-num-shards 1 2 2
```

To sweep worker settings:

```bash
torchrun-hpc -N 4 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark \
  -c $(pwd)/ScaFFold/configs/benchmark_testing.yml \
  --problem-scale 7 --dc-num-shards 1 2 2 \
  --dataloader-num-workers 2 \
  --dataloader-prefetch-factor 4
```

## Acceptance Criteria

- DataLoader construction reflects the configured worker and prefetch values.
- No per-sample `glob` remains in `__getitem__`.
- Mask remapping uses the lookup-table fast path for the normal integer-mask dataset.
- Epoch `2+` timing improves or stays flat for the required comparison matrix.
