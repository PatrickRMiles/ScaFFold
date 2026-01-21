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

        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

        customlog(
            f"Creating dataset with {len(self.ids)} examples. Loading from {data_dir}"
        )
        with open(data_dir, "rb") as data_file:
            data = pickle.load(data_file)
        self.mask_values = data["mask_values"]
        customlog(f"Unique mask values: {self.mask_values}")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, img, is_mask):
        if is_mask:
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
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert len(img_file) == 1, (
            f"Either no image or multiple images found for the ID {name}: {img_file}"
        )
        assert len(mask_file) == 1, (
            f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        )
        with open(mask_file[0], "rb") as f:
            mask = np.load(f)
        f.close()
        with open(img_file[0], "rb") as f:
            img = np.load(f)
        f.close()

        img = self.preprocess(self.mask_values, img, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, is_mask=True)

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
        }


class FractalDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, data_dir):
        super().__init__(images_dir, mask_dir, mask_suffix="_mask", data_dir=data_dir)
