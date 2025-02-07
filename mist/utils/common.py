import os
import inspect
import logging
import json
import pickle
import random
from functools import partial
from pprint import pprint
import subprocess
import psutil
import torch.distributed
from tqdm import tqdm
from time import time, perf_counter
from typing import Callable, Optional, Sequence, Tuple

from filelock import FileLock

import numpy as np

import torch


# 1. dtype utils
# 2. file io utils
# 3. benchmark utils

KB = 1024
MB = 1024**2
GB = 1024**3

# ======================================================================================
# Randomness
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


# ======================================================================================
# dtype utils
def str_to_torch_dtype(dtype_str):
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "half": torch.half,
        "float": torch.float,
        "double": torch.double,
        "int": torch.int,
        "long": torch.long,
        "short": torch.short,
    }
    assert (
        dtype_str in mapping
    ), f"Unknown dtype {dtype_str}. Supported dtypes: {tuple(mapping.keys())}"
    return mapping[dtype_str]


def torch_dtype_to_str(dtype):
    mapping = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.int16: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.half: "float16",
        torch.float: "float32",
        torch.double: "float64",
        torch.int: "int32",
        torch.long: "int64",
        torch.short: "int16",
    }
    assert (
        dtype in mapping
    ), f"Unknown dtype {dtype}. Supported dtypes: {tuple(mapping.keys())}"
    return mapping[dtype]


# ======================================================================================
# file io utils
def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def save_json(data, filename, mode="w"):
    with open(filename, mode) as f:
        _string = json.dumps(data, indent=2)
        print(_string, file=f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_json_with_lock(filename, logging_level=None):
    if logging_level is not None:
        assert logging_level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL] 
        logging.getLogger("filelock").setLevel(logging_level)

    assert filename.endswith(".json")
    lock_path = filename + ".lock"
    lock = FileLock(lock_path)
    with lock:
        with open(filename, "r") as f:
            return json.load(f)

def save_json_with_lock(filename, update_fn: Callable, logging_level=None):
    if logging_level is not None:
        assert logging_level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL] 
        logging.getLogger("filelock").setLevel(logging_level)

    assert filename.endswith(".json")
    lock_path = filename + ".lock"
    lock = FileLock(lock_path)
    if os.path.exists(filename):
        with lock:
            with open(filename, "r+") as f:
                data = json.load(f)
                _data = update_fn(data)
                data = _data if _data is not None else data
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
    else:
        data = {}
        _data = update_fn(data)
        data = _data if _data is not None else data
        with lock:
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)


def print_to_file(*args, mode="a", filename=None, to_screen=True, **kwargs):
    assert filename is not None or to_screen
    if filename is not None:
        with open(filename, mode=mode) as f:
            print(*args, file=f, **kwargs)
    if to_screen:
        print(*args, **kwargs)


def pprint_to_file(*args, mode="a", filename=None, to_screen=True, **kwargs):
    assert filename is not None or to_screen
    sort_dicts = kwargs.pop("sort_dicts", False)
    indent = kwargs.pop("indent", 1)
    if filename is not None:
        with open(filename, mode=mode) as f:
            pprint(*args, stream=f, indent=indent, sort_dicts=sort_dicts, **kwargs)
    if to_screen:
        pprint(*args, indent=indent, sort_dicts=sort_dicts, **kwargs)


def write_tsv(heads, values, filename: str, logging_fn=None):
    """Write tsv data to a file."""
    assert len(heads) == len(values), f"Got {len(heads)} heads and {len(values)} values"
    values = [str(x) for x in values]

    with open(filename, "a", encoding="utf-8") as fout:
        fout.write("\t".join(values) + "\n")

    if logging_fn is not None:
        line = ""
        for i in range(len(heads)):
            line += heads[i] + ": " + values[i] + "  "
        logging_fn(line)


# ======================================================================================
# Utils for benchmarking
def benchmark_func_walltime(
    func,
    warmup,
    number,
    prepare_func=None,
    sync_func=None,
    num_memory_records: int = 0,
    enable_tqdm=False,
):
    sync_func = sync_func or (lambda: None)

    peak_allocated_memories = []
    peak_reserved_memories = []
    torch.cuda.reset_peak_memory_stats()

    def _run_func():
        # Generate inputs if prepare_func is provided
        inputs = prepare_func() if prepare_func is not None else None

        # Run the function and do synchronization
        sync_func()
        tic = perf_counter()
        if inputs is not None:
            args, kwargs = inputs
            func(*args, **kwargs)
        else:
            func()
        sync_func()
        toc = perf_counter()

        if len(peak_allocated_memories) < num_memory_records:
            peak_allocated_memories.append(torch.cuda.max_memory_allocated())
            peak_reserved_memories.append(torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()

        # Return the cost
        return toc - tic

    # Warmup
    for _ in tqdm(range(warmup), disable=not enable_tqdm):
        _run_func()

    costs = []
    # Benchmark
    for _ in tqdm(range(number), disable=not enable_tqdm):
        cost = _run_func()
        costs.append(cost)

    peak_allocated_memories = np.array(peak_allocated_memories) / MB
    peak_allocated_memories = np.sort(np.unique(peak_allocated_memories))
    peak_reserved_memories = np.array(peak_reserved_memories) / MB
    peak_reserved_memories = np.sort(np.unique(peak_reserved_memories))

    return np.array(costs), (peak_allocated_memories, peak_reserved_memories)


def benchmark_func_cuda_event(
    func: Callable,
    warmup: int,
    number: int,
    prepare_func: Callable = None,
    sync_func: Callable = None,
    num_memory_records: int = 0,
    enable_tqdm: bool = False,
):
    sync_func = sync_func or (lambda: None)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(number + 1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(number + 1)]

    peak_allocated_memories = []
    peak_reserved_memories = []
    torch.cuda.reset_peak_memory_stats()

    def _run_func(step=-1):
        start_event = start_events[step]
        end_event = end_events[step]

        # Generate inputs if prepare_func is provided
        inputs = prepare_func() if prepare_func is not None else None

        # Run the function and do synchronization
        # Run the forward
        start_event.record()
        if inputs is not None:
            assert isinstance(inputs, tuple) and len(inputs) == 2
            args, kwargs = inputs
            func(*args, **kwargs)
        else:
            func()
        end_event.record()

        if len(peak_allocated_memories) < num_memory_records:
            peak_allocated_memories.append(torch.cuda.max_memory_allocated())
            peak_reserved_memories.append(torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in tqdm(range(warmup), disable=not enable_tqdm):
        _run_func()

    # Benchmark
    for step in tqdm(range(number), disable=not enable_tqdm):
        _run_func(step)

    torch.cuda.synchronize()
    costs = [
        start_events[i].elapsed_time(end_events[i]) / 1000.0 for i in range(number)
    ]

    peak_allocated_memories = np.array(peak_allocated_memories) / MB
    peak_allocated_memories = np.sort(np.unique(peak_allocated_memories))
    peak_reserved_memories = np.array(peak_reserved_memories) / MB
    peak_reserved_memories = np.sort(np.unique(peak_reserved_memories))

    return np.array(costs), (peak_allocated_memories, peak_reserved_memories)


def benchmark_func_for_fwd_bwd_walltime(
    fwd_func,
    warmup,
    number,
    prepare_func=None,
    sync_func=None,
    num_memory_records=0,
    enable_tqdm=False,
):
    sync_func = sync_func or (lambda: None)

    fwd_peak_memories = []
    bwd_peak_memories = []
    torch.cuda.reset_peak_memory_stats()

    def _run_func():
        # Generate inputs if prepare_func is provided
        inputs = prepare_func() if prepare_func is not None else None

        # Run the function and do synchronization
        # Run the forward
        sync_func()
        tic_fwd = perf_counter()
        if inputs is not None:
            assert (
                isinstance(inputs, tuple) and len(inputs) == 2
            ), f"Should be (args, kwargs), but got {inputs}"
            args, kwargs = inputs
            output = fwd_func(*args, **kwargs)
        else:
            output = fwd_func()
        sync_func()
        toc_fwd = perf_counter()

        # Memory
        if len(fwd_peak_memories) < num_memory_records:
            fwd_peak_memories.append(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()

        # Calculate the dummy loss
        loss = output.sum()

        # Run the backward
        sync_func()
        tic_bwd = perf_counter()
        loss.backward()
        sync_func()
        toc_bwd = perf_counter()

        # Memory
        if len(bwd_peak_memories) < num_memory_records:
            bwd_peak_memories.append(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()

        # Return the cost
        return toc_fwd - tic_fwd, toc_bwd - tic_bwd

    # Warmup
    for _ in tqdm(range(warmup), disable=not enable_tqdm):
        _run_func()

    costs_fwd, costs_bwd = [], []
    # Benchmark
    for _ in tqdm(range(number), disable=not enable_tqdm):
        cost_fwd, cost_bwd = _run_func()
        costs_fwd.append(cost_fwd)
        costs_bwd.append(cost_bwd)

    return (
        np.array(costs_fwd),
        np.array(costs_bwd),
        (fwd_peak_memories, bwd_peak_memories),
    )


def benchmark_func_for_fwd_bwd_opt_walltime(
    fwd_func,
    warmup,
    number,
    opt_func,
    gradient_accumulation_steps=1,
    prepare_func=None,
    sync_func=None,
    post_fwd_func=None,
    post_bwd_func=None,
    num_memory_records=0,
    enable_tqdm=False,
):
    sync_func = sync_func or (lambda: None)

    fwd_peak_allocated_memories = []
    bwd_peak_allocated_memories = []
    opt_peak_allocated_memories = []
    fwd_peak_reserved_memories = []
    bwd_peak_reserved_memories = []
    opt_peak_reserved_memories = []
    torch.cuda.reset_peak_memory_stats()

    def _run_func():
        fwd_time = 0
        bwd_time = 0
        for i in range(gradient_accumulation_steps):
            # Generate inputs if prepare_func is provided
            inputs = prepare_func() if prepare_func is not None else None

            # Run the function and do synchronization
            # Run the forward
            sync_func()
            tic_fwd = perf_counter()
            if inputs is not None:
                assert (
                    isinstance(inputs, tuple) and len(inputs) == 2
                ), f"Should be (args, kwargs), but got {inputs}"
                args, kwargs = inputs
                loss = fwd_func(*args, **kwargs)
            else:
                loss = fwd_func()
            sync_func()
            toc_fwd = perf_counter()
            fwd_time += toc_fwd - tic_fwd

            # Memory
            if (
                len(fwd_peak_allocated_memories)
                < num_memory_records * gradient_accumulation_steps
            ):
                fwd_peak_allocated_memories.append(torch.cuda.max_memory_allocated())
                fwd_peak_reserved_memories.append(torch.cuda.max_memory_reserved())
                torch.cuda.reset_peak_memory_stats()

            if post_fwd_func is not None:
                post_fwd_func()

            # Run the backward
            sync_func()
            tic_bwd = perf_counter()
            loss.backward()
            sync_func()
            toc_bwd = perf_counter()
            bwd_time += toc_bwd - tic_bwd

            # Memory
            if (
                len(bwd_peak_allocated_memories)
                < num_memory_records * gradient_accumulation_steps
            ):
                bwd_peak_allocated_memories.append(torch.cuda.max_memory_allocated())
                bwd_peak_reserved_memories.append(torch.cuda.max_memory_reserved())
                torch.cuda.reset_peak_memory_stats()

            if post_bwd_func is not None:
                post_bwd_func()

            if i == gradient_accumulation_steps - 1:
                # Run the optimizer
                sync_func()
                tic_opt = perf_counter()
                if opt_func is not None:
                    opt_func()
                sync_func()
                toc_opt = perf_counter()

                # Memory
                if len(opt_peak_allocated_memories) < num_memory_records:
                    opt_peak_allocated_memories.append(
                        torch.cuda.max_memory_allocated()
                    )
                    opt_peak_reserved_memories.append(torch.cuda.max_memory_reserved())
                    torch.cuda.reset_peak_memory_stats()

        # Return the cost
        return (
            fwd_time / gradient_accumulation_steps,
            bwd_time / gradient_accumulation_steps,
            toc_opt - tic_opt,
        )

    # Warmup
    for _ in tqdm(range(warmup), disable=not enable_tqdm):
        _run_func()

    costs_fwd, costs_bwd, costs_opt = [], [], []
    # Benchmark
    for _ in tqdm(range(number), disable=not enable_tqdm):
        cost_fwd, cost_bwd, cost_opt = _run_func()
        costs_fwd.append(cost_fwd)
        costs_bwd.append(cost_bwd)
        costs_opt.append(cost_opt)

    fwd_peak_allocated_memories = np.array(fwd_peak_allocated_memories)
    bwd_peak_allocated_memories = np.array(bwd_peak_allocated_memories)
    opt_peak_allocated_memories = np.array(opt_peak_allocated_memories)
    fwd_peak_reserved_memories = np.array(fwd_peak_reserved_memories)
    bwd_peak_reserved_memories = np.array(bwd_peak_reserved_memories)
    opt_peak_reserved_memories = np.array(opt_peak_reserved_memories)

    fwd_peak_allocated_memories = np.sort(np.unique(fwd_peak_allocated_memories)) / MB
    bwd_peak_allocated_memories = np.sort(np.unique(bwd_peak_allocated_memories)) / MB
    opt_peak_allocated_memories = np.sort(np.unique(opt_peak_allocated_memories)) / MB
    fwd_peak_reserved_memories = np.sort(np.unique(fwd_peak_reserved_memories)) / MB
    bwd_peak_reserved_memories = np.sort(np.unique(bwd_peak_reserved_memories)) / MB
    opt_peak_reserved_memories = np.sort(np.unique(opt_peak_reserved_memories)) / MB

    return (
        np.array(costs_fwd),
        np.array(costs_bwd),
        np.array(costs_opt),
        (fwd_peak_allocated_memories, fwd_peak_reserved_memories),
        (bwd_peak_allocated_memories, bwd_peak_reserved_memories),
        (opt_peak_allocated_memories, opt_peak_reserved_memories),
    )


def _format_time(t):
    # By default, time is in seconds, convert to us, ms, s
    t = t * 1e6
    prefixes = ["us", "ms", "s"]
    prefix = prefixes[0]
    for new_prefix in prefixes[1:]:
        if t < 1000:
            break
        prefix = new_prefix
        t /= 1000
    return "{:.4f} {}".format(t, prefix)


def process_benchmarking_results(
    costs: np.array,
    msg: str = "",
    print_to_screen: bool = False,
    print_fn: Callable = print,
) -> Tuple[float, float, float]:
    """Process the benchmarking results.

    Parameters
    ----------
    costs
        the array of the costs
    msg, optional
        message to show, by default ""
    print_to_screen, optional
        whether to print to the screen, by default False
    print_fn, optional
        print function, by default print

    Returns
    -------
    mean
        the mean of the costs
    median
        the median of the costs
    std
        the standard deviation of the costs
    """
    mean = costs.mean()
    median = np.median(costs)
    std = costs.std()
    if print_to_screen:
        msg = f"[{msg}]\t" if msg else ""
        print_fn(
            f"{msg}Median: {_format_time(median)}, Mean: {_format_time(mean)}, Std: {std:.8f}"
        )
    return mean, median, std


def log_memory(
    allocated_memories: Optional[Sequence[int]] = None,
    reserved_memories: Optional[Sequence[int]] = None,
    msg: str = "",
    get_peak_if_memories_not_provided: bool = True,
    reset_memory_stats: bool = False,
    print_fn: Callable = print,
):
    """Log the allocated and reserved memories.

    Parameters
    ----------
    allocated_memories
        the sequence of the allocated memory, by default None
    reserved_memories
        the sequence of the reserved memory, by default None
    msg
        message to show, by default ""
    get_peak
        get peak memory if memory sequence is not provided, by default True
    reset_memory_stats
        reset peak memory stats, by default False
    print_fn
        print function, by default print
    """
    msg = f"[{msg}]\t" if msg else ""

    if allocated_memories is not None:
        assert isinstance(allocated_memories, (list, tuple, np.ndarray))
    elif get_peak_if_memories_not_provided:
        allocated_memories = [torch.cuda.max_memory_allocated() / MB]
    else:
        allocated_memories = [torch.cuda.memory_allocated() / MB]

    if reserved_memories is not None:
        assert isinstance(reserved_memories, (list, tuple, np.ndarray))
    elif get_peak_if_memories_not_provided:
        reserved_memories = [torch.cuda.max_memory_reserved() / MB]
    else:
        reserved_memories = [torch.cuda.memory_reserved() / MB]

    if reset_memory_stats:
        torch.cuda.reset_peak_memory_stats()

    allocated_memories_str = ", ".join([f"{mem:.2f}" for mem in allocated_memories])
    reserved_memories_str = ", ".join([f"{mem:.2f}" for mem in reserved_memories])
    print_fn(f"{msg}Allocated memories: [{allocated_memories_str}] MB")
    print_fn(f"{msg}Reserved  memories: [{reserved_memories_str}] MB")


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


# ======================================================================================
# psutil utils
def find_and_kill_other_processes(pattern: str, workers_for_pdsh: Optional[str]=None):
    # Elegant way to kill other processes
    # matching_processes = []
    # for proc in psutil.process_iter(["pid", "name", "cmdline"]):
    #     try:
    #         if proc.info["cmdline"] is None:
    #             continue
    #         if pattern in proc.info["cmdline"]:
    #             matching_processes.append(proc)
    #     except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
    #         pass

    # for proc in matching_processes:
    #     if proc.pid != os.getpid():
    #         print(f"Killing process {proc.pid}")
    #         proc.kill()

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        num_nodes = world_size // torch.cuda.device_count()
        if num_nodes > 1:
            print(f"Killing other processing using ** pdsh -f 1024 -R ssh -w {workers_for_pdsh} 'pkill -2 -f {pattern} && sleep 1' ** ", flush=True)
            os.system(
                f"pdsh -f 1024 -R ssh -w {workers_for_pdsh} 'pkill -2 -f {pattern} && sleep 3'",
            )
    os.system(f"pkill -2 -f {pattern} && sleep 3")
