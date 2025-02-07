from typing import Optional, List, Tuple, Any, Dict, Union
import argparse
import os
import random
import time
import math
from numbers import Number
from itertools import product
from tqdm import tqdm
from functools import lru_cache

from einops import rearrange
import numpy as np
import torch
import torch.distributed as dist

from mist.utils.common import (
    benchmark_func_cuda_event,
    benchmark_func_walltime,
    load_json,
    process_benchmarking_results,
    save_json,
)
from mist.utils.device import get_simplified_device_name
from mist.utils.memory import cuda_empty_cache

# These environment variables can potentially improve the performance
# when the CPU is weak for a PCIe-based system. However, this may cause
# hangs for some cases.
# os.environ["NCCL_SHM_USE_CUDA_MEMCPY"] = "1"
# os.environ["NCCL_CREATE_THREAD_CONTEXT"] = "1"

# =====================================================================
TFLOPS_REFS = {
    "a10g": 31.4,
    "l4": 30.29,
    "v100-sxm2-32gb": 125,
    "v100-sxm2-16gb": 125,
    "a100-sxm4-40gb": 312,
}

G2G_BANDWIDTH_REFS = {
    "a10g": 6.0,
    "l4": 6.0,
    "v100-sxm2-32gb": 120,
    "v100-sxm2-16gb": 120,
    "a100-sxm4-40gb": 240,
}

C2G_BANDWIDTH_REFS = {
    "a10g": 10,
    "l4": 6,
    "v100-sxm2-32gb": 6,
    "v100-sxm2-16gb": 3,
    "a100-sxm4-40gb": 5,
}

MEM_CAPACITY = torch.cuda.get_device_properties(0).total_memory
LRU_CACHE_SIZE = math.ceil(0.6 * MEM_CAPACITY / 1024 ** 3)

def get_reference_compute_time(device_name: str, n: int, m: int, k: int) -> float:
    tflops = TFLOPS_REFS[device_name]
    return (n * m * k) / (tflops * 1e12)


def get_reference_g2g_time(device_name: str, mbytes: int, nnodes: int) -> float:
    if nnodes == 1:
        return mbytes / (G2G_BANDWIDTH_REFS[device_name] * 1024)
    else:
        return mbytes / (6 * 1024)


def get_reference_c2g_or_g2c_time(device_name: str, mbytes: int) -> float:
    return mbytes / (C2G_BANDWIDTH_REFS[device_name] * 1024)


def round_up_to_multiple_of(x: Number, v=8) -> int:
    return int((x + v - 1) // v * v)


# =====================================================================


device_count = torch.cuda.device_count()

try:
    dist.init_process_group(backend="nccl")
    dist_initialized = True
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
except:
    dist_initialized = False
    local_rank = 0
    global_rank = 0
    world_size = 1

# Process group will be initialized later
process_group = None

num_nodes = max(world_size // device_count, 1)
num_gpus_per_node = min(world_size, device_count)

dtype = torch.float16
element_size = dtype.itemsize
device = torch.device("cuda", local_rank)
device_name = get_simplified_device_name(device, lower=True)
warmup = 20
number = 10

async_op = False
all_gather_stream = torch.cuda.Stream(priority=-1)
cpu_to_gpu_stream = torch.cuda.Stream(priority=-1)
gpu_to_cpu_stream = torch.cuda.Stream(priority=-1)
compute_stream = torch.cuda.Stream(priority=-1)

# Base hyperparameters
COMPUTE_M = 4096
COMPUTE_K = 4096
MB = 1024**2
BASE_TIME_MSEC = 3
RATIOS = [0, 1, 2, 3, 4]


@lru_cache(maxsize=LRU_CACHE_SIZE)
def get_tensor(
    type_: str,
    size: Tuple[int],
    dtype: torch.dtype,
    device: torch.device,
    requires_grad=False,
    **kwargs,
):
    if type_ == "zeros":
        return torch.zeros(
            size, dtype=dtype, device=device, requires_grad=requires_grad, **kwargs
        )
    elif type_ == "ones":
        return torch.ones(
            size, dtype=dtype, device=device, requires_grad=requires_grad, **kwargs
        )
    elif type_ == "rand":
        ret = torch.randint(0, 10, size, dtype=dtype, device=device, **kwargs) / 100
        ret.requires_grad_(requires_grad)
        return ret
    else:
        raise ValueError(f"Invalid type: {type_}")


def benchmark_one_case(
    results: dict,
    compute_n: int,
    gpu_to_gpu_mbytes: int,
    cpu_to_gpu_mbytes: int,
    gpu_to_cpu_mbytes: int,
    args: argparse.Namespace,
    profile: bool = False,
    print_to_stdout: bool = True,
):
    compute_m = COMPUTE_M
    compute_k = COMPUTE_K
    gpu_to_gpu_numel = gpu_to_gpu_mbytes * MB // element_size
    cpu_to_gpu_numel = cpu_to_gpu_mbytes * MB // element_size
    gpu_to_cpu_numel = gpu_to_cpu_mbytes * MB // element_size
    case_name = f"N_{compute_n}-M_{compute_m}-K_{compute_k}-G2G_{gpu_to_gpu_mbytes}-C2G_{cpu_to_gpu_mbytes}-G2C_{gpu_to_cpu_mbytes}-W_{world_size}-Intra_{args.intra_group_size}-Inter_{args.inter_group_size}"

    # Write the results if the case is already in the cache
    if (
        results is not None
        and (device_name in results and case_name in results[device_name])
        and not profile
    ):
        if local_rank == 0:
            print(f"Hit the cache: {case_name}")
        return False

    # All gather input and output
    # all_gather_input = torch.zeros(
    #     gpu_to_gpu_numel // process_group.size(), dtype=dtype, device=device
    # )
    # all_gather_output = torch.zeros(gpu_to_gpu_numel, dtype=dtype, device=device)
    all_gather_input = get_tensor(
        "rand", (gpu_to_gpu_numel // process_group.size(),), dtype, device
    )
    all_gather_output = get_tensor("zeros", (gpu_to_gpu_numel,), dtype, device)

    # CPU to GPU copy
    # a_cpu_tensor = torch.zeros(
    #     cpu_to_gpu_numel,
    #     dtype=dtype,
    #     device="cpu",
    #     requires_grad=False,
    #     pin_memory=True,
    # )
    # a_gpu_tensor = torch.zeros(
    #     cpu_to_gpu_numel, dtype=dtype, device=device, requires_grad=False
    # )
    a_cpu_tensor = get_tensor(
        "rand",
        (cpu_to_gpu_numel,),
        dtype,
        torch.device("cpu"),
        requires_grad=False,
        pin_memory=True,
    )
    a_gpu_tensor = get_tensor(
        "zeros", (cpu_to_gpu_numel,), dtype, device, requires_grad=False
    )

    # GPU to CPU copy
    # b_cpu_tensor = torch.zeros(
    #     gpu_to_cpu_numel,
    #     dtype=dtype,
    #     device="cpu",
    #     requires_grad=False,
    #     pin_memory=True,
    # )
    # b_gpu_tensor = torch.zeros(
    #     gpu_to_cpu_numel, dtype=dtype, device=device, requires_grad=False
    # )
    b_cpu_tensor = get_tensor(
        "rand",
        (gpu_to_cpu_numel,),
        dtype,
        torch.device("cpu"),
        requires_grad=False,
        pin_memory=True,
    )
    b_gpu_tensor = get_tensor(
        "zeros", (gpu_to_cpu_numel,), dtype, device, requires_grad=False
    )

    # Compute
    mat_a = get_tensor("ones", (compute_n, compute_k), dtype, device)
    mat_b = None
    mat_out = None
    if compute_n > 0:
        mat_b = get_tensor("ones", (compute_k, compute_m), dtype, device)
        mat_out = get_tensor("zeros", (compute_n, compute_m), dtype, device)

    def compute():
        if compute_n <= 0:
            return
        with torch.no_grad():
            with torch.cuda.stream(compute_stream):
                torch.matmul(mat_a, mat_b, out=mat_out)

    def all_gather():
        if gpu_to_gpu_numel <= 0 or process_group is None:
            return
        with torch.cuda.stream(all_gather_stream):
            dist.all_gather_into_tensor(
                all_gather_output,
                all_gather_input,
                group=process_group,
                async_op=async_op,
            )

    def cpu_to_gpu_copy():
        if cpu_to_gpu_numel <= 0:
            return
        with torch.cuda.stream(cpu_to_gpu_stream):
            a_gpu_tensor.copy_(a_cpu_tensor, non_blocking=True)

    def gpu_to_cpu_copy():
        if gpu_to_cpu_numel <= 0:
            return
        with torch.cuda.stream(gpu_to_cpu_stream):
            b_cpu_tensor.copy_(b_gpu_tensor, non_blocking=True)

    inner_iters = 1

    def run_func():
        for _ in range(inner_iters):
            gpu_to_cpu_copy()
            cpu_to_gpu_copy()
            all_gather()
            compute()

    def sync_func():
        torch.cuda.synchronize()
        if dist_initialized:
            dist.barrier()
            torch.cuda.synchronize()

    costs, _ = benchmark_func_walltime(
        run_func,
        warmup=warmup,
        number=number,
        sync_func=sync_func,
        enable_tqdm=False,
    )

    median_cost = np.mean(costs) / inner_iters
    std = np.std(costs) / inner_iters
    compute_gflops = (compute_n * compute_m * compute_k) / (median_cost * 1e9)
    g2g_bandwidth = (gpu_to_gpu_mbytes / 1024) / median_cost
    c2g_bandwidth = (cpu_to_gpu_mbytes / 1024) / median_cost
    g2c_bandwidth = (gpu_to_cpu_mbytes / 1024) / median_cost
    if local_rank == 0 and print_to_stdout:
        print(f"Case {case_name}:")
        print(f" - End-to-end: {median_cost * 1e3:.2f} ms")
        print(
            f" - {compute_gflops=:.2f} GFLOPS, {g2g_bandwidth=:.2f} GB/s, {c2g_bandwidth=:.2f} GB/s, {g2c_bandwidth=:.2f} GB/s"
        )
        print(
            f" - STD: {np.std(costs) * 1e3:.4f}; STD/Mean: {std / median_cost * 100:.4f}%"
        )
        if std / median_cost > 0.1:
            print(costs)

    # Update the results
    if results is not None:
        info = {
            "compute_n": str(compute_n),
            "compute_m": str(compute_m),
            "compute_k": str(compute_k),
            "gpu_to_gpu_mbytes": str(gpu_to_gpu_mbytes),
            "cpu_to_gpu_mbytes": str(cpu_to_gpu_mbytes),
            "gpu_to_cpu_mbytes": str(gpu_to_cpu_mbytes),
            "latency": f"{median_cost * 1e3:.2f} ms",
            "std": f"{std * 1e3:.4f} ms",
            "std/mean": f"{std / median_cost:.4f}",
            "world_size": str(world_size),
            "intra_group_size": str(args.intra_group_size),
            "inter_group_size": str(args.inter_group_size),
            "gflops": f"{compute_gflops:.2f} GFLOPS",
            "g2g_bandwidth": f"{g2g_bandwidth:.2f} GB/s",
            "c2g_bandwidth": f"{c2g_bandwidth:.2f} GB/s",
            "g2c_bandwidth": f"{g2c_bandwidth:.2f} GB/s",
        }
        results.setdefault(device_name, {})[case_name] = info

    # ==============================================================================
    # Profile
    profile_warmup = 15
    profile_number = 5
    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0, warmup=profile_warmup, active=profile_number
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
            record_shapes=True,
            # with_stack=True,
            # with_modules=True,
        ) as profiler:
            for i in range(profile_warmup + profile_number):
                run_func()
                sync_func()
                profiler.step()
    # ==============================================================================

    # Clean up memory if reserving too much
    
    if torch.cuda.memory_reserved() > 0.85 * MEM_CAPACITY:
        cuda_empty_cache()

    return True


def benchmark(args):
    ONLY_COMM_MBYTES = list(32 * n for n in range(2, 17))
    # Get the reference values
    compute_ref_time = get_reference_compute_time(
        device_name, n=1, m=COMPUTE_M, k=COMPUTE_K
    )
    g2g_ref_time = get_reference_g2g_time(device_name, mbytes=1, nnodes=num_nodes)
    c2g_ref_time = get_reference_c2g_or_g2c_time(device_name, mbytes=1)

    # Get the benchmarking values
    compute_n_candidates = []
    g2g_mbytes_candidates = []
    c2g_mbytes_candidates = []
    g2c_mbytes_candidates = []
    for ratio in RATIOS:
        sec = BASE_TIME_MSEC * ratio * 1e-3
        compute_n_candidates.append(round_up_to_multiple_of(sec / compute_ref_time))
        g2g_mbytes_candidates.append(round_up_to_multiple_of(sec / g2g_ref_time))
        c2g_mbytes_candidates.append((round_up_to_multiple_of(sec / c2g_ref_time)))
        g2c_mbytes_candidates.append((round_up_to_multiple_of(sec / c2g_ref_time)))

    # Print the candidates
    print(f"compute_n_candidates: {compute_n_candidates}")
    print(f"g2g_mbytes_candidates: {g2g_mbytes_candidates}")
    print(f"c2g_mbytes_candidates: {c2g_mbytes_candidates}")
    print(f"g2c_mbytes_candidates: {g2c_mbytes_candidates}")

    # Collect choices
    choices = []
    for only_comm_scale in ONLY_COMM_MBYTES:
        choices.append((0, only_comm_scale, 0, 0))
        choices.append((0, 0, only_comm_scale, 0))
        choices.append((0, 0, 0, only_comm_scale))
    if not args.only_comm_and_no_overlap:
        raw_choices = list(
            product(
                compute_n_candidates,
                g2g_mbytes_candidates,
                c2g_mbytes_candidates,
                g2c_mbytes_candidates,
            )
        )
        for raw in set(raw_choices):
            b, g2g, c2g, g2c = raw
            if b == 0 and g2g == 0 and c2g == 0 and g2c == 0:
                continue
            choices.append(raw)
    choices = list(set(choices))
    choices = sorted(choices, key=lambda x: sum(x[1:]))
    choices = list(reversed(choices))
    print(f"Number of choices: {len(choices)}")

    # Get the output_path
    output_dir = args.output
    output_path = os.path.join(
        output_dir,
        f"bandwidth-{get_simplified_device_name(lower=True)}-n{num_nodes}-m{num_gpus_per_node}-intra{args.intra_group_size}-inter{args.inter_group_size}.json",
    )

    # Load the results
    if not os.path.exists(output_dir):
        if local_rank == 0:
            os.makedirs(output_dir)
    if dist_initialized:
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
    if os.path.exists(output_path):
        results = load_json(output_path)
    else:
        results = {device_name: {}}

    # Warmup
    n_before_all_warmp = min(15, len(choices))
    warmup_choices = choices[:n_before_all_warmp]
    for i, (n, g2g, c2g, g2c) in enumerate(tqdm(warmup_choices)):
        benchmark_one_case(
            None,
            n,
            g2g,
            c2g,
            g2c,
            args=args,
            profile=False,
        )
        time.sleep(0.1)

    # Run the cases
    for i, (n, g2g, c2g, g2c) in enumerate(tqdm(choices)):
        worked = benchmark_one_case(
            results,
            n,
            g2g,
            c2g,
            g2c,
            args=args,
            profile=False,
        )
        if worked:
            time.sleep(0.1)
        if local_rank == 0 and i % 50 == 0:
            save_json(results, output_path)

    torch.cuda.synchronize()
    dist.barrier()
    if local_rank == 0:
        save_json(results, output_path)


def single_case(args):
    for i in range(20):
        benchmark_one_case(
            None,
            args.compute_n,
            args.gpu_to_gpu_mbytes,
            args.cpu_to_gpu_mbytes,
            args.gpu_to_cpu_mbytes,
            args=args,
            profile=args.profile,
            print_to_stdout=False
        )

    benchmark_one_case(
        None,
        args.compute_n,
        args.gpu_to_gpu_mbytes,
        args.cpu_to_gpu_mbytes,
        args.gpu_to_cpu_mbytes,
        args=args,
        profile=args.profile,
    )


def verify_and_update_args(args):
    assert (
        num_gpus_per_node % args.intra_group_size == 0
    ), f"Invalid intra group size: {args.intra_group_size} does not divide {num_gpus_per_node}"
    assert (
        num_nodes % args.inter_group_size == 0
    ), f"Invalid inter group size: {args.inter_group_size} does not divide {num_nodes}"
    if device_name in [
        "v100-sxm2-32gb",
        "v100-sxm2-16gb",
        "a100-sxm4-40gb",
        "a100-sxm4-80gb",
    ]:
        args.nvlink = True
    return args


def init_process_group(args):
    """Initialize the process groups based on the intra and inter group sizes."""
    global process_group

    if (
        args.intra_group_size == num_gpus_per_node
        and args.inter_group_size == num_nodes
    ):
        process_group = dist.group.WORLD
    else:
        ranks = np.arange(world_size).reshape(num_nodes, num_gpus_per_node)
        ranks_per_group = rearrange(
            ranks,
            "(h1 a) (w1 b) -> (h1 w1) (a b)",
            a=args.inter_group_size,
            b=args.intra_group_size,
        )
        for group_idx in range(ranks_per_group.shape[0]):
            tmp_ranks = ranks_per_group[group_idx]
            tmp_group = dist.new_group(tmp_ranks)
            if global_rank in tmp_ranks:
                process_group = tmp_group
            if global_rank == tmp_ranks[0]:
                print(f"Group {group_idx}: {tmp_ranks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile GPU to CPU and CPU to GPU bandwidth"
    )
    parser.add_argument(
        "--single-case",
        "-s",
        action="store_true",
        help="Whether to run a single case",
    )
    parser.add_argument(
        "--only-comm-and-no-overlap",
        "-oc",
        action="store_true",
        help="Whether to only profile the communication",
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Whether to profile the communication",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/",
        help="The output json file to save the results",
    )
    parser.add_argument(
        "--intra-group-size",
        type=int,
        default=num_gpus_per_node,
        help="The number of groups for the communication",
    )
    parser.add_argument(
        "--inter-group-size",
        type=int,
        default=num_nodes,
        help="The number of groups for the communication",
    )
    parser.add_argument(
        "--compute-n",
        "-c",
        type=int,
        default=0,
        help="The compute n for the matrix multiplication",
    )
    parser.add_argument(
        "--cpu-to-gpu-mbytes",
        "-c2g",
        type=int,
        default=0,
        help="The mbytes for the CPU to GPU communication",
    )
    parser.add_argument(
        "--gpu-to-cpu-mbytes",
        "-g2c",
        type=int,
        default=0,
        help="The scale for the GPU to CPU communication",
    )
    parser.add_argument(
        "--gpu-to-gpu-mbytes",
        "-g2g",
        type=int,
        default=0,
        help="The scale for the GPU to GPU communication",
    )
    parser.add_argument(
        "--nvlink",
        action="store_true",
        help="Whether to use the nvlink",
    )
    args = parser.parse_args()
    verify_and_update_args(args)
    init_process_group(args)
    if args.single_case:
        single_case(args)
    else:
        benchmark(args)
