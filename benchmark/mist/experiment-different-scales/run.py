import argparse
import os
import time
from itertools import product
from typing import List, Optional, Union, Tuple, Any, Dict, Callable
import json
from omegaconf import DictConfig, OmegaConf
import warnings

import torch

from mist.utils.device import get_simplified_device_name, mock_cuda_device_name_if_needed

TEMPLATES = {
    "a10g": "template-a10g",
    "l4": "template-l4",
    "v100-sxm2-16gb": "template-v100-sxm2-16gb",
    "v100-sxm2-32gb": "template-v100-sxm2-32gb",
    "a100-sxm4-40gb": "template-a100-sxm4-40gb",
}
def get_template():
    device_name = get_simplified_device_name(torch.device("cuda"), lower=True)
    template = TEMPLATES[device_name]
    return template


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


# =====================================================================
# Collect all possible configurations
# =====================================================================
TUNING_GRANULARITY_CHOICES = [
    "no-pp",
    "uniform-pp",
    "uniform-device-pp",
    "uniform-device-pp-mip",
]
# STATE_OFFLOADING_CHOICES = [True, False]
# ACTIVATION_OFFLOADING_CHOICES = [True, False]
# OFFLOADING_CHOICES is (STATE_OFFLOADING, ACTIVATION_OFFLOADING)
# OFFLOADING_CHOICES = [(True, True), (True, False), (False, True), (False, False)]
# OFFLOADING_CHOICES = [(True, True), (True, False), (False, True)]
# OFFLOADING_CHOICES = [(False, False)]  # For now, only the default configuration for debugging
# =====================================================================
# tuning_granularity_choices = ["uniform-device-pp-mip"]
# ckpt_tuning_choices = [True]
# offloading_choices = OFFLOADING_CHOICES
# choices = []
# choices = list(
#     product(tuning_granularity_choices, ckpt_tuning_choices, offloading_choices)
# )
# for tuning_granularity_choice in tuning_granularity_choices:
#     choices.append((tuning_granularity_choice, False, (False, False)))
# choices = list(set(choices))

choices = [
    # Tuning granularity, zero-2/3, ckpt_tuning, (state_offloading, activation_offloading), flex_pipe
    ("uniform-device-pp-mip", False, False, (False, False), 1),
    ("uniform-device-pp-mip", True, False, (False, False), 1),
    ("uniform-device-pp-mip", True, True, (False, False), 1),
    ("uniform-device-pp-mip", True, True, (True, False), 30),
    # ("uniform-device-pp-mip", True, True, (False, True), 30),
    ("uniform-device-pp-mip", False, True, (True, True), 30),
    ("uniform-device-pp-mip", True, True, (True, True), 30),
]

# =====================================================================


def tune_and_analyze_all(args):
    # bw_str = ""
    # if args.inter_bw is not None:
    #     bw_str +=
    # if args.intra_bw is not None:
    #     bw_str += f"hardware.intra_node_gpu_gpu_bandwidth={args.intra_bw} "
    # if args.cpu_bw is not None:
    #     bw_str += f"hardware.gpu_cpu_bandwidth={args.cpu_bw} "

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
        state_offloading, activation_offloading = offloading_choice
        config_file_name = f"config-{tuning_granularity}-zero23_{zero_23}-ckpt_{ckpt_tuning}-so_{state_offloading}-ao_{activation_offloading}-ss_{sample_size}"
        output_path = os.path.join(output_dir, config_file_name)
        log_file_path = os.path.join(output_dir, f"log-{config_file_name}.log")

        # =====================================================================
        # Skip - Temporary
        # Skip if the configuration is already executed
        output_file_path = output_path + ".yaml"
        if os.path.exists(output_file_path):
            summary_file_path = os.path.join(output_dir, "summary.json")
            if os.path.exists(summary_file_path):
                summary = load_json(summary_file_path)
                if config_file_name in summary:
                    data = summary[config_file_name]
                    if "exec_total_cost" in data:
                        print(f"Configuration {config_file_name} is already executed.")
                        continue
        # =====================================================================

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
            f"output_path={output_path} "
            f">> {log_file_path} 2>&1"
        )
        run_cmd(f"echo {tuning_cmd} >> {log_file_path}", print_cmd=False)
        run_cmd(tuning_cmd)
        run_cmd(f"echo '' >> {log_file_path}", print_cmd=False)
        time.sleep(0.2)

        # Run analysis
        analyzing_cmd = (
            "python ../analysis/run.py "
            f"-cd {output_dir} "
            f"--config-name {config_file_name}"
            f">> {log_file_path} 2>&1"
        )
        run_cmd(f"echo {analyzing_cmd} >> {log_file_path}", print_cmd=False)
        run_cmd(analyzing_cmd)
        run_cmd(f"echo '' >> {log_file_path}", print_cmd=False)
        time.sleep(0.2)


def exec_all(args):
    """
    Exec all unique configurations.
    """
    output_dir = get_output_dir(args)

    # Get the file mapping from the summary.json
    summary = load_json(os.path.join(output_dir, "summary.json"))
    unique_solutions: Dict[str, str] = {}
    for config_name, config_summary in summary.items():
        solution = config_summary["tuning_best_solution"]
        unique_solutions.setdefault(solution, []).append(config_name)

    # Map the configurations
    config_mapping: Dict[str, List[str]] = {}
    for configs_sharing_the_same_solution in unique_solutions.values():
        first_solution = configs_sharing_the_same_solution[0]
        config_mapping[first_solution] = configs_sharing_the_same_solution[1:]

    for (
        tuning_granularity,
        zero_23,
        ckpt_tuning,
        offloading_choice,
        sample_size,
    ) in sorted(choices):
        state_offloading, activation_offloading = offloading_choice
        config_file_name = f"config-{tuning_granularity}-zero23_{zero_23}-ckpt_{ckpt_tuning}-so_{state_offloading}-ao_{activation_offloading}-ss_{sample_size}"
        if config_file_name not in config_mapping:
            continue  # Skip the configurations because the solution is executed already

        output_path = os.path.join(output_dir, config_file_name + ".yaml")
        if not os.path.exists(output_path):
            warnings.warn(f"Configuration {config_file_name} does not exist.")
            continue

        log_file_path = os.path.join(output_dir, f"log-{config_file_name}.log")

        # =====================================================================
        # Skip - Temporary
        if os.path.exists(os.path.join(output_dir, "summary.json")):
            summary = load_json(os.path.join(output_dir, "summary.json"))
            if config_file_name in summary:
                if "exec_total_cost" in summary[config_file_name]:
                    print(f"Configuration {config_file_name} is already executed.")
                    # continue
            else:
                # Used to manually skip the configurations
                print(f"Configuration {config_file_name} is not in the summary.")
                continue
        # =====================================================================

        # Run execution
        if args.nnodes == 1:
            exec_cmd = (
                "torchrun "
                f"--nproc_per_node={args.nproc_per_node} "
                f"../exec/benchmark_one_case.py "
                f"-cd {output_dir} "
                f"--config-name {config_file_name} "
                "profile=False "
                "tiny_bench=True "
                f">> {log_file_path} 2>&1"
            )
            run_cmd(f"echo {exec_cmd} >> {log_file_path}", print_cmd=False)
            run_cmd(exec_cmd)
            run_cmd(f"echo '' >> {log_file_path}", print_cmd=False)
            time.sleep(0.5)

        else:
            exec_cmd = (
                f"pdsh -S -f 1024 -R ssh -w worker-[1-{args.nnodes}] '"
                f"cd {os.getcwd()} && "
                f"{args.pdsh_init_cmd} && "
                f"NCCL_ASYNC_ERROR_HANDLING=1 "
                f"torchrun "
                f"--nnodes={args.nnodes} "
                f"--nproc_per_node={args.nproc_per_node} "
                f"--node_rank={args.node_rank} "
                f"--master_addr={args.master_addr} "
                f"--master_port={args.master_port} "
                f"../exec/benchmark_one_case.py "
                f"-cd {output_dir} "
                f"--config-name {config_file_name} "
                "profile=False "
                "tiny_bench=True "
                f">> {log_file_path} 2>&1"
                "'"
            )
            run_cmd(f"echo {exec_cmd} >> {log_file_path}", print_cmd=False)
            run_cmd(exec_cmd)
            run_cmd(f"echo '' >> {log_file_path}", print_cmd=False)
            time.sleep(1.0)

        # Update the summary for other configurations sharing the same solution
        summary = load_json(os.path.join(output_dir, "summary.json"))
        keys = [
            "exec_total_cost",
            "exec_stage_peak_allocated_memories",
            "exec_stage_peak_reserved_memories",
        ]
        if keys[0] not in summary[config_file_name]:
            continue
        updated_info = {key: summary[config_file_name][key] for key in keys}
        for other_config_file_name in config_mapping[config_file_name]:
            summary[other_config_file_name].update(updated_info)
        save_json(summary, os.path.join(output_dir, "summary.json"))


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
    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--skip-exec", action="store_true")

    args = parser.parse_args()

    with mock_cuda_device_name_if_needed():
        update_args(args)

        if not args.skip_tune:
            tune_and_analyze_all(args)
        if not args.skip_exec:
            exec_all(args)
