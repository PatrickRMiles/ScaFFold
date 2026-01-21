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
from distconv import DCTensor
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from tqdm import tqdm

from ScaFFold.utils.dice_score import dice_coeff, dice_loss, multiclass_dice_coeff
from ScaFFold.utils.perf_measure import annotate


@annotate()
@torch.inference_mode()
def evaluate(
    net, dataloader, device, amp, primary, criterion, n_categories, parallel_strategy
):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0.0
    processed_batches = 0

    # For reference, dc sharding happens on this spatial dim: 2=D, 3=H, 4=W
    if primary:
        print(
            f"[eval] ps.shard_dim={parallel_strategy.shard_dim} num_shards={parallel_strategy.num_shards}"
        )

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        val_loss_epoch = 0.0
        for batch in tqdm(
            dataloader,
            total=num_val_batches,
            desc="Validation round",
            unit="batch",
            leave=False,
            disable=not primary,
        ):
            image, mask_true = batch["image"], batch["mask"]

            # move images and labels to correct device and type
            image = image.to(
                device=device,
                dtype=torch.float32,
                memory_format=torch.channels_last_3d,  # NDHWC (channels last) vs NCDHW (channels first)
            )
            mask_true = mask_true.to(
                device=device, dtype=torch.long
            ).contiguous()  # masks no channels NDHW, but ensure cotinuity.

            # Shard batch across ddp mesh, replicate across dc mesh
            image_dp = distribute_tensor(
                image, parallel_strategy.device_mesh, placements=[Shard(0), Replicate()]
            ).to_local()
            mask_true_dp = distribute_tensor(
                mask_true,
                parallel_strategy.device_mesh,
                placements=[Shard(0), Replicate()],
            ).to_local()

            # Spatially shard images along the dc mesh and run the model
            dcx = DCTensor.distribute(image_dp, parallel_strategy)
            dcy = net(dcx)

            # Replicate predictions across dc to get full spatial result on each dc rank
            mask_pred = dcy.to_replicate()

            # Use labels that are replicated across dc and sharded across ddp, like predictions
            mask_true_ddp = mask_true_dp

            # Skip if this ddp rank has an empty local batch
            if mask_pred.size(0) == 0 or mask_true_ddp.size(0) == 0:
                continue

            # Loss
            CE_loss = criterion(mask_pred, mask_true_ddp)

            # Dice loss
            mask_pred_softmax = F.softmax(mask_pred, dim=1).float()
            mask_true_onehot = (
                F.one_hot(mask_true_ddp, n_categories + 1)
                .permute(0, 4, 1, 2, 3)
                .float()
            )
            dice_loss_curr = dice_loss(
                mask_pred_softmax,
                mask_true_onehot,
                multiclass=True,
            )

            # Combined validation loss
            loss = CE_loss + dice_loss_curr
            val_loss_epoch += loss.item()
            processed_batches += 1

            # Dice score
            if net.module.n_classes == 1:
                assert mask_true_ddp.min() >= 0 and mask_true_ddp.max() <= 1, (
                    "True mask indices should be in [0, 1]"
                )
                mask_pred_bin = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(
                    mask_pred_bin, mask_true_ddp, reduce_batch_first=False
                )
            else:
                assert (
                    mask_true_ddp.min() >= 0
                    and mask_true_ddp.max() < net.module.n_classes
                ), "True mask indices should be in [0, n_classes]"
                mask_pred_processed = F.softmax(mask_pred, dim=1).float()
                mask_true_onehot_mc = (
                    F.one_hot(mask_true_ddp, net.module.n_classes)
                    .permute(0, 4, 1, 2, 3)
                    .float()
                )
                dice_score += multiclass_dice_coeff(
                    mask_pred_processed[:, 1:],
                    mask_true_onehot_mc[:, 1:],
                    reduce_batch_first=True,
                )

    net.train()

    val_loss_avg = val_loss_epoch / max(processed_batches, 1)
    if primary:
        print(
            f"evaluate.py: dice_score={dice_score}, val_loss_epoch={val_loss_epoch}, val_loss_avg={val_loss_avg}, num_val_batches={processed_batches}"
        )
    return dice_score, val_loss_epoch, val_loss_avg, processed_batches
