"""
This script is used to profile the bandwidth of the network within a single node.
1. Intra-node bandwidth
2. CPU-GPU bandwidth

And we want to test the bandwidth of the following settings:
1. Raw (no other computation)
2. Overlapped (with computation)
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist

import random

from itertools import product

from mist.utils.common import (
    run_cmd,
    load_json,
    save_json,
)
from tqdm import tqdm

BASE_ELEMENTS = 2048 * 2048 * 3
SCALES = [0, 1, 2, 4, 8]

raw_choices = list(product(SCALES, SCALES, SCALES))
choices = []
for raw in set(raw_choices):
    if raw[0] == 0 and raw[1] == 0 and raw[2] == 0:
        continue
    if raw[1] > raw[2]:
        continue
    choices.append(raw)
choices = sorted(choices)
print(f"Number of choices: {len(choices)}")


def run_cases(args):
    for g2g_scale, c2g_scale, g2c_scale in tqdm(choices):
        g2g = BASE_ELEMENTS * g2g_scale
        c2g = BASE_ELEMENTS * c2g_scale
        g2c = BASE_ELEMENTS * g2c_scale
        for g in range(2, args.nproc_per_node + 1):
            cmd = (
                "torchrun "
                f"--nproc_per_node {g} "
                "_interference_estimation_one_case.py "
                f"-g2g {g2g} "
                f"-c2g {c2g} "
                f"-g2c {g2c} "
            )
            run_cmd(cmd)
            time.sleep(0.5)

    # for b, s, h in product(BATCH_SIZES, SEQ_LENGTHS, HIDDEN_DIMS):
    #     shape = f"({b}, {s}, {h})"
    #     for g in range(2, args.nproc_per_node + 1):
    #         cmd = (
    #             "torchrun "
    #             f"--nproc_per_node {g} "
    #             "_profile_single_node_bandwidth_one_case.py "
    #             f"--shape '{shape}' "
    #         )
    #         run_cmd(cmd)
    #         time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", "-n", type=int, default=1)
    parser.add_argument("--nproc_per_node", "-m", type=int, default=4)
    args = parser.parse_args()
    run_cases(args)
