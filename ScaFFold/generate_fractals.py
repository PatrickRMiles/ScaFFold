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

from argparse import Namespace

from mpi4py import MPI

from ScaFFold.datagen import category_search, instance


def main(kwargs_dict: dict = {}):
    args = Namespace(**kwargs_dict)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f"generate_fractals.py: world size = {size}")

    comm.Barrier()

    category_search.main(args)

    comm.Barrier()

    instance.main(args)

    comm.Barrier()

    if rank == 0:
        print(
            f"generate_fractals.py({rank}): Fractal and instance generation has finished. Exiting..."
        )

    MPI.Finalize()

    return 0
