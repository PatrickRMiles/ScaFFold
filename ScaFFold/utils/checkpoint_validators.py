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

import os

# import pdb

if hasattr(os, "sched_getaffinity"):
    _orig_affinity = os.sched_getaffinity(0)
else:
    _orig_affinity = None

import torch


def compare_state_dicts(quantity: str, dict1, dict2):
    def _compare(dict1, dict2, prefix=""):
        equal = True
        if dict1.keys() != dict2.keys():
            missing_in_dict2 = dict1.keys() - dict2.keys()
            missing_in_dict1 = dict2.keys() - dict1.keys()
            if missing_in_dict2:
                print(
                    f"train.py: {quantity} missing in dict2: {', '.join(missing_in_dict2)} at {prefix}"
                )
                equal = False
            if missing_in_dict1:
                print(
                    f"train.py: {quantity} missing in dict1: {', '.join(missing_in_dict1)} at {prefix}"
                )
                equal = False

        for key in dict1.keys() & dict2.keys():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(dict1[key], torch.Tensor) and isinstance(
                dict2[key], torch.Tensor
            ):
                if not torch.equal(dict1[key], dict2[key]):
                    print(
                        f"train.py: {quantity} tensor discrepancy at {full_key}: dict1[{key}] != dict2[{key}]"
                    )
                    equal = False
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                if not _compare(dict1[key], dict2[key], prefix=full_key):
                    equal = False
            else:
                if dict1[key] != dict2[key]:
                    print(
                        f"train.py: {quantity} value discrepancy at {full_key}: dict1[{key}]={dict1[key]}, dict2[{key}]={dict2[key]}"
                    )
                    equal = False
        return equal

    return _compare(dict1, dict2)


def compare_state_dicts2(*dicts):
    keys = dicts[0].keys()
    for key in keys:
        values = [d[key] for d in dicts]
        tensor_comparisons = [
            (
                torch.equal(values[i], values[i + 1])
                if torch.is_tensor(values[i])
                else values[i] == values[i + 1]
            )
            for i in range(len(values) - 1)
        ]
        if not all(tensor_comparisons):
            return False
    return True


def compare_tensors(tensor1, tensor2):
    return torch.all(torch.eq(tensor1, tensor2))


def compare_items(item1, item2):
    if isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
        return compare_tensors(item1, item2)
    elif isinstance(item1, dict) and isinstance(item2, dict):
        return compare_dicts3(item1, item2)
    else:
        return item1 == item2


def compare_dicts3(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1.keys():
        if not compare_items(dict1[key], dict2[key]):
            return False
    return True


#
# Usage in `train.py` below:
#

# For debugging, write the saved optimizer state to file to compare to loaded state on restart
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# optim_saved_path = f"{dir_checkpoint}/restarts/optim_saved_epoch{epoch}_{timestamp}.txt"
# with open(optim_saved_path, "w") as optim_f:
#     optim_f.write(str(state_dict['optimizer_state_dict']))

#
# For debugging purposes, load that checkpoint into new model, optimizer, etc and compare to active
#
# prev_checkpoint = torch.load(checkpoint_path)
# newmodel = UNet(n_channels=3, n_classes=n_classes, trilinear=False, layers=unet_layers)
# newmodel = newmodel.to(memory_format=torch.channels_last_3d)
# newmodel.to(device=device)
# newmodel = torch.nn.parallel.DistributedDataParallel(newmodel, device_ids=[get_cuda_device()], output_device=get_cuda_device())
# newmodel.module.load_state_dict(prev_checkpoint['model_state_dict'])
# newoptimizer = optim.RMSprop(newmodel.parameters(),
#                 lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
# if optimizer_name == "ADAM":
#     print(f"train.py(w{rank}|l{local_rank}): using ADAM optimizer .........")
#     newoptimizer = optim.Adam(newmodel.parameters(), lr=learning_rate)
# elif optimizer_name == "SGD":
#     print(f"train.py(w{rank}|l{local_rank}): using SGD optimizer .........")
#     newoptimizer = optim.SGD(newmodel.parameters(), lr=learning_rate, momentum=0.9)
# newoptimizer.load_state_dict(prev_checkpoint['optimizer_state_dict'])
# newscheduler = optim.lr_scheduler.ReduceLROnPlateau(newoptimizer, 'max', patience=25)
# newscheduler.load_state_dict(prev_checkpoint['scheduler_state_dict'])

# # Compare model state dicts
# model_compare = compare_state_dicts("model", model.state_dict(), newmodel.state_dict())
# optimizer_compare = compare_state_dicts("optimizer", optimizer.state_dict(), newoptimizer.state_dict())
# scheduler_compare = compare_state_dicts("scheduler", scheduler.state_dict(), newscheduler.state_dict())
# print(f"train.py: model_compare={model_compare}, optimizer_compare={optimizer_compare}, scheduler_compare={scheduler_compare}")
# print(f"train.py: all equal? {all(compare_dicts3(sd, optimizer.state_dict()) for sd in [state_dict['optimizer_state_dict'], prev_checkpoint['optimizer_state_dict'], newoptimizer.state_dict()])}")
