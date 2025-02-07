"""
Usage:
1. original:
    torchrun --nproc_per_node=2 main.py [-p]
2. sharding:
    torchrun --nproc_per_node=2 main.py -sh [-p]
3. swap:
    torchrun --nproc_per_node=2 main.py -e [-p] [--wsr 1.0] [--gsr 1.0] [--asr 1.0]
4. sharding + swap:
    torchrun --nproc_per_node=2 main.py -sh -e [-p] [--wsr 1.0] [--gsr 1.0] [--asr 1.0]

Note:
1. `-p` is for profiling.
2. use `--disable_overlap` to disable the overlap of the swapped tensors.
"""
import argparse
from typing import List, Optional, Union, Tuple, Any, Dict, Callable

import sympy as sp

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
from torch.optim import AdamW
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from mist.logger import get_logger, update_logger_formatter_for_rank
from mist.re_swap_manager.flat_param import HandleShardingStrategy
from mist.re_swap_manager.manager import ModelReSwapManager
from mist.utils.initialization import init_empty_weights
from mist.utils.common import (
    benchmark_func_for_fwd_bwd_walltime,
    benchmark_func_for_fwd_bwd_opt_walltime,
    process_benchmarking_results,
)

logger = get_logger()
logger.setLevel("INFO")

# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--hidden_size", "-d", type=int, default=2048)
parser.add_argument("--num_layers", "-n", type=int, default=8)
parser.add_argument("--batch_size", "-b", type=int, default=8)
parser.add_argument("--seq_len", "-s", type=int, default=2048)
parser.add_argument("--weight_swap_ratio", "--wsr", type=float, default=1.0)
parser.add_argument("--grad_swap_ratio", "--gsr", type=float, default=1.0)
parser.add_argument("--optimizer_swap_ratio", "--osr", type=float, default=1.0)
parser.add_argument("--activation_swap_ratio", "--asr", type=float, default=1.0)
parser.add_argument("--enable_sharding", "-sh", action="store_true")
parser.add_argument("--enable_swap", "-e", action="store_true")
parser.add_argument("--use_optim", "-o", action="store_true")
parser.add_argument("--cpu_optim", "--co", action="store_true")
parser.add_argument("--profile", "-p", action="store_true")
parser.add_argument("--disable_overlap", action="store_true")

# ==============================================================================


def create_block(hidden_size):
    return torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size, bias=False),
        torch.nn.Linear(hidden_size, hidden_size, bias=False),
        torch.nn.Linear(hidden_size, hidden_size * 4, bias=False),
        torch.nn.Linear(hidden_size * 4, hidden_size, bias=False),
    )


class DummyModel(torch.nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [create_block(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def main(args):
    device_offset = 2

    # Initialize the configs
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank + device_offset)
    device = torch.device("cuda", torch.cuda.current_device())
    process_group = dist.group.WORLD

    # Update the logger for the rank
    update_logger_formatter_for_rank(logger, disable_print=rank != 0)
    logger.info(f"rank: {rank}, world_size: {world_size}, device: {device}")

    # Construct a meta model to immitate the distributed training case
    model = DummyModel(args.hidden_size, args.num_layers)
    for name, module in model.named_modules():
        module.name = name

    # Construct the model re-swap manager
    overlapped_pairs = [
        (curr_module.name, next_module.name)
        for curr_module, next_module in zip(model.blocks[:-1], model.blocks[1:])
    ]
    overlapped_pairs.append((None, model.blocks[0].name))
    overlapped_pairs.append((model.blocks[-1].name, None))
    sharding_strategy = (
        HandleShardingStrategy.FULL_SHARD
        if args.enable_sharding
        else HandleShardingStrategy.NO_SHARD
    )
    enable_re_swap = args.enable_swap or args.enable_sharding
    if not args.enable_swap:
        args.weight_swap_ratio = 0.0
        args.grad_swap_ratio = 0.0
        args.optimizer_swap_ratio = 0.0
        args.activation_swap_ratio = 0.0
    model_re_swap_manager = ModelReSwapManager(
        overlapped_pairs=overlapped_pairs,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        cuda_device=device,
        overlap=not args.disable_overlap,
    )
    # ==============================================================================
    # Debugging
    # enable_re_swap = True
    # ==============================================================================
    for i, block in enumerate(model.blocks):
        is_first = i == 0
        is_last = i == len(model.blocks) - 1
        block.cuda(device)
        # The overhead of applying the re-swap manager even nothing is enabled is
        # negligible. e.g. 1.01%.
        model_re_swap_manager.apply(
            block,
            state_swap_ratio=(args.weight_swap_ratio, args.grad_swap_ratio),
            activation_swap_ratio=args.activation_swap_ratio,
            device=torch.cuda.current_device(),
            is_first=is_first,
            is_last=is_last,
        )
    if enable_re_swap:
        model_re_swap_manager.register_hooks()
    torch.cuda.synchronize()

    # ==============================================================================
    # Run - correctness test
    # ==============================================================================
    # for i in range(10):
    #     hidden_states = torch.rand(
    #         args.batch_size,
    #         args.seq_len,
    #         args.hidden_size,
    #         device=device,
    #         requires_grad=True,
    #     )
    #     outputs = model(hidden_states)
    #     loss = outputs.sum()
    #     loss.backward()

    #     for block in model.blocks:
    #         flat_param_handle_group = model_re_swap_manager.flat_param_handles[
    #             block.name
    #         ]
    #         handle = flat_param_handle_group.handles()[0]
    #         flat_param = handle.flat_param
    #         for param in flat_param._params:
    #             assert param.grad is None
    #         assert flat_param.grad is None
    #         assert flat_param._saved_grad_shard is not None

    # ###########
    # Debugging #
    # ###########
    # for name, param in model.named_parameters():
    #     logger.info(f"{name}: [ID]: {id(param)}, [DATAPTR]: {param.data_ptr()} [Shape]: {param.shape}, [StorageSize]: {param._typed_storage()._size()}, [Device]: {param.device}")

    warmup = 1 if args.profile else 10
    number = 1 if args.profile else 10

    def prepare_func():
        hidden_states = torch.rand(
            args.batch_size,
            args.seq_len,
            args.hidden_size,
            device=device,
            requires_grad=True,
        )
        return (hidden_states,), {}

    def fwd_func(hidden_states):
        return model(hidden_states)

    if args.use_optim:
        if not args.enable_sharding and not args.enable_swap:
            optimizer = AdamW(model.parameters(), lr=1e-3)
        else:
            raise NotImplementedError(
                "Optimizer is not supported for sharding and swap"
            )

        def opt_func():
            optimizer.step()

    else:

        def opt_func():
            pass

    (
        fwd_latencies,
        bwd_latencies,
        opt_latencies,
        (fwd_peak_memories, bwd_peak_memories, opt_peak_memories),
    ) = benchmark_func_for_fwd_bwd_opt_walltime(
        fwd_func,
        warmup=warmup,
        number=number,
        opt_func=opt_func,
        prepare_func=prepare_func,
        sync_func=torch.cuda.synchronize,
        num_memory_records=10,
        enable_tqdm=True,
    )
    logger.info(f"FWD peak memories: {fwd_peak_memories[:2]}")
    logger.info(f"BWD peak memories: {bwd_peak_memories[:2]}")
    logger.info(f"OPT peak memories: {opt_peak_memories[:2]}")

    (
        fwd_latencies_mean,
        fwd_latencies_median,
        fwd_latencies_std,
    ) = process_benchmarking_results(
        fwd_latencies, msg="Forward", print_to_screen=True, logger=logger
    )
    (
        bwd_latencies_mean,
        bwd_latencies_median,
        bwd_latencies_std,
    ) = process_benchmarking_results(
        bwd_latencies, msg="Backward", print_to_screen=True, logger=logger
    )

    (
        opt_latencies_mean,
        opt_latencies_median,
        opt_latencies_std,
    ) = process_benchmarking_results(
        opt_latencies, msg="OptStep", print_to_screen=True, logger=logger
    )

    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
            record_shapes=True,
            # with_stack=True,
            # with_modules=True,
        ) as profiler:
            for i in range(7):
                _args, _kwargs = prepare_func()
                output = fwd_func(*_args, **_kwargs)
                torch.cuda.synchronize()
                loss = output.sum()
                loss.backward()
                torch.cuda.synchronize()

                profiler.step()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
