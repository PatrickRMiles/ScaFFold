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


def detect_scheduler():
    """Detects which scheduler is being used based on environment variables."""
    if "FLUX_JOB_ID" in os.environ or "FLUX_JOB_NNODES" in os.environ:
        return "flux"
    elif "SLURM_JOB_ID" in os.environ or "SLURM_NNODES" in os.environ:
        return "slurm"
    else:
        return "unknown"


def collect_flux_metadata():
    """Collects metadata from Flux environment variables."""
    metadata = {
        "scheduler": "flux",
        "job_id": os.environ.get("FLUX_JOB_ID"),
        "num_nodes": os.environ.get("FLUX_JOB_NNODES"),
        "num_tasks": os.environ.get("FLUX_JOB_SIZE"),
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        metadata["visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    elif os.environ.get("ROCR_VISIBLE_DEVICES") is not None:
        metadata["visible_devices"] = os.environ.get("ROCR_VISIBLE_DEVICES")
    else:
        metadata["visible_devices"] = ""
    return metadata


def collect_slurm_metadata():
    """Collects metadata from Slurm environment variables."""
    metadata = {
        "scheduler": "slurm",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "num_nodes": os.environ.get("SLURM_NNODES"),
        "num_tasks": os.environ.get("SLURM_NTASKS"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        metadata["visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    elif os.environ.get("ROCR_VISIBLE_DEVICES") is not None:
        metadata["visible_devices"] = os.environ.get("ROCR_VISIBLE_DEVICES")
    else:
        metadata["visible_devices"] = ""
    return metadata


def collect_scheduler_metadata():
    """Detects scheduler and collects the relevant metadata."""
    scheduler = detect_scheduler()
    if scheduler == "flux":
        metadata = collect_flux_metadata()
    elif scheduler == "slurm":
        metadata = collect_slurm_metadata()
    else:
        metadata = {"scheduler": "unknown"}
    return {k: v for k, v in metadata.items()}
