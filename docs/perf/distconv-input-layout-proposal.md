# DistConv Input-Layout Proposal

This document captures the deferred DistConv/input-layout optimization work so it can be picked up in a separate implementation chat.

## Current Hot Path

The warmup, training, and evaluation loops all repeat the same input-layout sequence for every batch:

1. Load dense local tensors from the `DataLoader`.
2. Move tensors to device.
3. Build a DTensor from the local tensor with batch sharding semantics.
4. Immediately call `.to_local()` on that DTensor.
5. Redistribute the resulting tensor with `DCTensor.distribute(...)`.

That sequence currently appears in all three paths:

- `PyTorchTrainer.warmup`
- `PyTorchTrainer.train`
- `evaluate`

The code also reconstructs logically identical placement metadata in the hot path, especially in evaluation.

## Why This Is Suspect

The saved traces show material host time in:

- `distconv.__torch_function__`
- `torch.overrides.handle_torch_function`
- DTensor mesh scatter/shard helpers
- repeated per-batch Python dispatch around layout conversion

That does not prove the layout conversion is the dominant bottleneck at all scales, but it is substantial enough to justify isolating in its own branch.

## Proposed Investigation Direction

The future branch should focus only on reducing repeated layout-conversion overhead without changing model math or sharding semantics.

Recommended implementation direction:

- Precompute and cache placement metadata once during trainer setup.
- Factor the shared dense-to-distconv conversion path into one helper used by warmup, train, and eval.
- Minimize repeated DTensor construction and redundant `.to_local()` transitions where DistConv can consume a cheaper equivalent form.
- Cache or precompute the flattened spatial process-group handle if the revised implementation still needs frequent collectives tied to the same spatial mesh.

## Non-Goals for That Branch

- No Dice math changes.
- No DataLoader changes.
- No optimizer or AMP changes.
- No change to the user-visible sharding interface.

## Evidence To Collect In The Future Branch

Before and after traces should specifically compare:

- total time in DTensor distribution helpers
- total time in DistConv tensor conversion helpers
- CPU gaps between `DataLoader` completion and the first major forward kernels
- steady-state epoch duration for:
  - `problem_scale=7`, `16` GPUs, `1 2 2`
  - `problem_scale=8`, `32` GPUs, `1 2 2`

## Risks

- DistConv may rely on the current handoff pattern for correctness even if it looks redundant.
- A cleaner helper extraction may reduce code duplication without producing a measurable speedup.
- The right optimization may depend on DistConv internals that are not obvious from the application code alone.
