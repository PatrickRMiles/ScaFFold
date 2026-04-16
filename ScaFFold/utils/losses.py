# Copyright (c) 2014-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LBANN/ScaFFold.
#
# SPDX-License-Identifier: (Apache-2.0)

import torch
import torch.nn.functional as F

from ScaFFold.utils.dice_score import SpatialAllReduce


def compute_sharded_cross_entropy_loss(
    local_preds,
    local_labels,
    spatial_mesh,
    _num_shards,
    device_type,
    class_weights=None,
):
    """
    Compute the CE loss for a spatially sharded volume.

    Each rank only sees a local spatial shard, so we cannot use the local
    `reduction="mean"` result directly. Instead we:
    1. compute the local CE numerator with `reduction="sum"`,
    2. build the correct global denominator,
    3. all-reduce across the spatial mesh, and
    4. divide to recover the same value we would get from a non-sharded tensor.

    When `class_weights` is provided, PyTorch's CE "mean" divides by the sum of
    the target weights, not the raw voxel count, so we reproduce that behavior
    explicitly here.
    """

    autocast_device = device_type if device_type != "mps" else "cpu"
    with torch.autocast(autocast_device, enabled=False):
        # Accumulate CE in full precision. Using reduction="sum" gives us the
        # numerator of the final global mean; if class weights are present,
        # PyTorch applies the target-class weight to each voxel here.
        local_ce_sum = F.cross_entropy(
            local_preds.float(),
            local_labels,
            weight=class_weights,
            reduction="sum",
        )

        if class_weights is None:
            # Sum the actual local voxel counts across spatial shards. We use
            # an all-reduced count instead of numel()*num_shards because shard
            # sizes can differ at chunk boundaries.
            local_voxel_count = local_ce_sum.new_tensor(float(local_labels.numel()))
            global_normalizer = SpatialAllReduce.apply(
                local_voxel_count, spatial_mesh
            )
        else:
            # Weighted CE divides by sum(weight[target_i]) over all voxels.
            # Build that denominator from the local label histogram, then
            # all-reduce it across the spatial mesh.
            local_class_counts = torch.bincount(
                local_labels.reshape(-1), minlength=class_weights.numel()
            ).to(dtype=local_ce_sum.dtype)
            local_weight_sum = torch.dot(
                local_class_counts, class_weights.to(dtype=local_ce_sum.dtype)
            )
            global_normalizer = SpatialAllReduce.apply(local_weight_sum, spatial_mesh)

    # Sum the local CE numerators from each spatial shard to get the global CE
    # numerator, then divide by the matching global denominator.
    global_ce_sum = SpatialAllReduce.apply(local_ce_sum, spatial_mesh)
    # Clamp to avoid a divide-by-zero in degenerate cases.
    return global_ce_sum / global_normalizer.clamp_min(
        torch.finfo(global_ce_sum.dtype).eps
    )
