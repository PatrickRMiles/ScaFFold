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
import torch.distributed as dist
from torch import Tensor

from ScaFFold.utils.perf_measure import annotate


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    assert input.size() == target.size(), (
        f"Got predictions (input) of {input.size()} and target of {target.size()}"
    )
    assert input.dim() == 4 or not reduce_batch_first

    sum_dim = (
        (-1, -2, -3) if input.dim() == 3 or not reduce_batch_first else (-1, -2, -3, -4)
    )

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum_raw = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum_raw == 0, inter, sets_sum_raw)

    dice = (inter + epsilon) / (sets_sum + epsilon)

    return dice.mean()


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all classes
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


@annotate()
def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class SpatialAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, reduce_group):
        output = input.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=reduce_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


@annotate()
def compute_sharded_dice(
    preds: torch.Tensor,
    labels: torch.Tensor,
    reduce_group,
    num_classes: int,
    epsilon: float = 1e-6,
):
    """
    Computes the globally sharded Dice score.
    Returns the raw score tensor of shape [Batch, Channels].
    """
    assert preds.dim() == 5, f"Expected 5D tensor, got {preds.dim()}D"
    assert labels.dim() == 4, f"Expected 4D labels tensor, got {labels.dim()}D"
    assert preds.size(0) == labels.size(0), (
        f"Batch mismatch: {preds.size(0)} vs {labels.size(0)}"
    )
    assert preds.shape[2:] == labels.shape[1:], (
        f"Spatial mismatch: {preds.shape} vs {labels.shape}"
    )

    batch_size = preds.size(0)
    preds_flat = preds.reshape(batch_size, num_classes, -1)
    labels_flat = labels.reshape(batch_size, -1).long()

    pred_sums = preds_flat.sum(dim=2)
    true_class_probs = preds_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1)

    intersections = torch.zeros_like(pred_sums)
    intersections.scatter_add_(1, labels_flat, true_class_probs)
    intersections.mul_(2.0)

    target_sums = torch.zeros_like(pred_sums)
    target_sums.scatter_add_(
        1, labels_flat, torch.ones_like(true_class_probs, dtype=preds.dtype)
    )

    packed = torch.stack([intersections, pred_sums + target_sums])

    # Global reduce across spatial mesh
    packed_global = SpatialAllReduce.apply(packed, reduce_group)

    global_inter = packed_global[0]
    global_sets_sum_raw = packed_global[1]

    global_sets_sum = torch.where(
        global_sets_sum_raw == 0, global_inter, global_sets_sum_raw
    )

    # Calculate score
    dice_score = (global_inter + epsilon) / (global_sets_sum + epsilon)

    return dice_score
