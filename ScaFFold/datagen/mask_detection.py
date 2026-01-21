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

import argparse
import logging
import pickle
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../data/masks/training",
        help="Where to check masks",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="../data/preprocess_data",
        help="Where to put the output",
    )
    args = parser.parse_args()
    return args


def unique_mask_values(idx, directory):
    mask_file = list(directory.glob(idx + ".*"))[0]
    with open(mask_file, "rb") as f:
        mask = np.load(f)
    f.close()
    # print(f'mask_detection.py: file {mask_file}, unique = {np.unique(mask)}')

    return np.unique(mask)


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    images_dir = Path(args.input)
    ids = [
        splitext(file)[0]
        for file in listdir(images_dir)
        if isfile(join(images_dir, file)) and not file.startswith(".")
    ]
    # print(f'mask_detection.py: ({len(ids)}) ids={ids}')
    if not ids:
        raise RuntimeError(
            f"No input file found in {images_dir}, make sure you put your images there"
        )

    logging.info(f"Scanning {len(ids)} masks")
    with Pool() as p:
        unique = list(
            tqdm(
                p.imap(partial(unique_mask_values, directory=images_dir), ids),
                total=len(ids),
            )
        )

    mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
    logging.info(f"Unique mask values: {mask_values}")

    # Saves the values in a pickle
    data = {"mask_values": mask_values}
    outfile = open(args.output, "wb")
    pickle.dump(data, outfile)
    outfile.close()
