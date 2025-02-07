import argparse
import os
import time
from itertools import product
from typing import List, Optional, Union, Tuple, Any, Dict, Callable
import json
from omegaconf import DictConfig, OmegaConf
import warnings

import torch

from mist.utils.device import (
    get_simplified_device_name,
    mock_cuda_device_name_if_needed,
)

def get_template():
    return "template-tuning-time"


# =====================================================================
# Utility functions
def run_cmd(cmd, print_cmd=True):
    if print_cmd:
        print(cmd, flush=True)
    os.system(cmd)


def save_json(data, filename, mode="w"):
    with open(filename, mode) as f:
        _string = json.dumps(data, indent=2)
        print(_string, file=f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


# =====================================================================


def get_output_dir(args):
    output_dir = (
        f"results/"
        f"{torch.cuda.get_device_name(0).replace(' ', '-').lower()}/"
        f"{args.model}-s_{args.seq_len}-v_{args.vocab_size}-b_{args.global_batch_size}-n_{args.nnodes}-m_{args.nproc_per_node}-f_{not args.disable_flash_attn}/"
        f"gpu_bw_{args._gpu_bw:.2f}-cpu_bw_{args._cpu_bw:.2f}/"
    )
    output_dir = os.path.abspath(output_dir)
    return output_dir


choices = [
    # Tuning granularity, zero-2/3, ckpt_tuning, (wo, go, oo, ao), flex_pipe
    # =====================================================================
    # For warmup
    ("uniform-device-pp-mip", False, False, (False, False, False, False), 1),
    # =====================================================================
    ("uniform-device-pp-mip", False, False, (False, False, False, False), 1),
    ("uniform-device-pp-mip", True, False, (False, False, False, False), 1),
    ("uniform-device-pp-mip", True, True, (False, False, False, False), 1),
    ("uniform-device-pp-mip", True, True, (False, False, True, False), 30),
    ("uniform-device-pp-mip", True, True, (False, False, True, True), 30), 
    ("uniform-device-pp-mip", True, True, (False, True, True, True), 30),
    ("uniform-device-pp-mip", True, True, (True, True, True, True), 30),
]

# =====================================================================


def tune_all(args):
    output_dir = get_output_dir(args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for (
        tuning_granularity,
        zero_23,
        ckpt_tuning,
        offloading_choice,
        sample_size,
    ) in choices:
        weight_offloading, grad_offloading, opt_state_offloading, activation_offloading = offloading_choice
        state_offloading = weight_offloading or grad_offloading or opt_state_offloading

        config_file_name = f"config-{tuning_granularity}-zero23_{zero_23}-ckpt_{ckpt_tuning}-wo_{weight_offloading}-go_{grad_offloading}-oo_{opt_state_offloading}-ao_{activation_offloading}-ss_{sample_size}"
        output_path = os.path.join(output_dir, config_file_name)
        log_file_path = os.path.join(output_dir, f"log-{config_file_name}.log")

        # Tune
        tuning_cmd = (
            "DISABLE_TQDM=true "
            "python ../tune/tune_one_case.py "
            f"experiment={get_template()} "
            f"model={args.model} "
            f"model.use_flash_attn={not args.disable_flash_attn} "
            f"training.max_sequence_length={args.seq_len} "
            f"training.vocab_size={args.vocab_size} "
            f"training.global_batch_size={args.global_batch_size} "
            f"hardware.num_nodes={args.nnodes} "
            f"hardware.num_gpus_per_node={args.nproc_per_node} "
            f"strategy.enabled=False "
            f"tuning.enabled=True "
            f"tuning.tuning_granularity={tuning_granularity} "
            f"tuning.zero_2_and_3_enabled={zero_23} "
            f"tuning.activation_checkpointing_tuning_enabled={ckpt_tuning} "
            f"tuning.state_offloading_enabled={state_offloading} "
            f"tuning.activation_offloading_enabled={activation_offloading} "
            f"tuning.sample_size={sample_size} "
            # =====================================================================
            # Added for the tuning time
            f"disable_wo_tuning={not weight_offloading} "
            f"disable_go_tuning={not grad_offloading} "
            f"disable_oo_tuning={not opt_state_offloading} "
            # =====================================================================
            f"output_path={output_path} "
            f">> {log_file_path} 2>&1"
        )
        start = time.time()
        run_cmd(f"echo {tuning_cmd} >> {log_file_path}", print_cmd=False)
        run_cmd(tuning_cmd)
        run_cmd(f"echo '' >> {log_file_path}", print_cmd=False)
        end = time.time()

        # Save the tuning time to the summary
        summary = load_json(os.path.join(output_dir, "summary.json"))
        summary[config_file_name].update({"tuning_time": end - start})
        save_json(summary, os.path.join(output_dir, "summary.json"), mode="w")

        time.sleep(1.0)

def update_args(args):
    if args.global_batch_size is None:
        args.global_batch_size = (
            args.global_batch_size_factor * args.nnodes * args.nproc_per_node
        )

    # Load yaml file
    yaml_file = f"../configs/experiment/{get_template()}.yaml"
    cfg = OmegaConf.load(yaml_file)

    args._gpu_bw = cfg.hardware.gpu_gpu_comm_params[3]
    args._cpu_bw = cfg.hardware.gpu_cpu_comm_params[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seq-len", "-s", type=int, default=2048)
    parser.add_argument("--vocab-size", "-v", type=int, default=50304)
    parser.add_argument("--global-batch-size", "-b", type=int, default=None)
    parser.add_argument("--global-batch-size-factor", "-f", type=int, default=16)
    parser.add_argument("--nnodes", "-n", type=int, default=1)
    parser.add_argument("--nproc-per-node", "-m", type=int, default=4)
    parser.add_argument("--master-addr", type=str, default="localhost")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--node-rank", type=str, default="0")
    parser.add_argument("--pdsh-init-cmd", type=str, default="echo ")
    parser.add_argument("--disable-flash-attn", action="store_true")
    parser.add_argument("--granularity", type=str, default="uniform-device-pp-mip")

    args = parser.parse_args()

    with mock_cuda_device_name_if_needed():
        update_args(args)
        tune_all(args)
