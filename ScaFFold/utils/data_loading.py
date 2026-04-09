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

import pickle
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ScaFFold.utils.utils import customlog


class BasicDataset(Dataset):
    def __init__(
        self, images_dir: str, mask_dir: str, mask_suffix: str = "", data_dir: str = ""
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix

        image_files = {
            splitext(file)[0]: self.images_dir / file
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        }
        mask_files = {}
        for file in listdir(mask_dir):
            if not isfile(join(mask_dir, file)) or file.startswith("."):
                continue
            mask_stem = splitext(file)[0]
            if not mask_stem.endswith(mask_suffix):
                continue
            sample_id = mask_stem[: -len(mask_suffix)]
            mask_files[sample_id] = self.mask_dir / file

        self.ids = sorted(set(image_files) & set(mask_files))
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )
        self.sample_paths = [
            (sample_id, image_files[sample_id], mask_files[sample_id])
            for sample_id in self.ids
        ]

        customlog(
            f"Creating dataset with {len(self.ids)} examples. Loading from {data_dir}"
        )
        with open(data_dir, "rb") as data_file:
            data = pickle.load(data_file)
        self.mask_values = data["mask_values"]
        self.mask_value_lookup = self._build_mask_value_lookup(self.mask_values)
        customlog(f"Unique mask values: {self.mask_values}")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _build_mask_value_lookup(mask_values):
        normalized_mask_values = [int(v) for v in mask_values]
        if normalized_mask_values == list(range(len(normalized_mask_values))):
            return np.arange(len(normalized_mask_values), dtype=np.int16)

        if min(normalized_mask_values, default=0) < 0:
            return None

        max_value = max(normalized_mask_values, default=0)
        lookup = np.full(max_value + 1, -1, dtype=np.int16)
        for idx, value in enumerate(normalized_mask_values):
            lookup[value] = idx
        return lookup

    @staticmethod
    def preprocess(mask_values, img, is_mask, mask_value_lookup=None):
        if is_mask:
            if (
                mask_value_lookup is not None
                and img.ndim == 3
                and np.issubdtype(img.dtype, np.integer)
                and img.min() >= 0
                and img.max() < len(mask_value_lookup)
            ):
                mapped_mask = mask_value_lookup[img]
                if (mapped_mask < 0).any():
                    raise ValueError(
                        "Encountered a mask value that was not present in mask_values"
                    )
                return mapped_mask

            mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.short)
            for i, v in enumerate(mask_values):
                if img.ndim == 3:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            img = img.transpose((3, 0, 1, 2))
            return img

    def __getitem__(self, idx):
        _, img_path, mask_path = self.sample_paths[idx]
        with open(mask_path, "rb") as f:
            mask = np.load(f)
        with open(img_path, "rb") as f:
            img = np.load(f)

        img = self.preprocess(self.mask_values, img, is_mask=False)
        mask = self.preprocess(
            self.mask_values,
            mask,
            is_mask=True,
            mask_value_lookup=self.mask_value_lookup,
        )

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
        }


class FractalDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, data_dir):
        super().__init__(images_dir, mask_dir, mask_suffix="_mask", data_dir=data_dir)
