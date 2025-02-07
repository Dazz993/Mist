import hydra
import inspect
import math
import os
import sys
import time
import traceback
import warnings
from collections import OrderedDict
from datetime import timedelta
from functools import partial
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List, Optional, Union, Tuple, Any, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.optim
from torch.cuda.amp import GradScaler

from apex.optimizers import FusedAdam

import mist
from mist import parallel_state
from mist.config import MistConfig
from mist.logger import get_logger, update_logger_formatter_for_rank
from mist.model_provider import base_model_provider, get_inputs_provider
from mist.overrides.base import reset_mist_patcher
from mist.optimizers.grad_scaler import ConstantGradScaler
from mist.pipeline_parallel.pipe_module import (
    build_pipe_modules_based_on_block_partition,
    PipeModule,
)
from mist.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
)
from mist.re_swap_manager.manager import ModelReSwapManager, HandleShardingStrategy
from mist.re_swap_manager.optimizer import ReSwapAdamW
from mist.tracer.symbolic_tracer import mist_trace, get_default_sub_modules
from mist.utils.common import (
    benchmark_func_cuda_event,
    benchmark_func_for_fwd_bwd_walltime,
    benchmark_func_for_fwd_bwd_opt_walltime,
    benchmark_func_walltime,
    log_memory,
    load_json,
    process_benchmarking_results,
    save_json,
    set_seed,
    find_and_kill_other_processes,
)
from mist.utils.device import all_params_and_buffers_in_device, get_device
from mist.utils.gradient_checkpointing import apply_gradient_checkpointing
from mist.utils.initialization import init_empty_weights
from mist.utils.memory import materialize_module, cuda_empty_cache
from mist.utils.module import count_module_parameters
from pprint import pformat

os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

logger = get_logger()

warnings.filterwarnings("ignore", category=UserWarning)

set_seed(12345)

_debug_count = 0

def _debug_print_model_weights_hash(model: nn.Module):
    global _debug_count
    for name, module in model.blocks.language_model.encoder.layers._modules.items():
        _sum = sum(p.float().sum().item() for p in module.parameters())
        logger.info(f"[ID {_debug_count}] [Layer {name}] sum: {_sum:.4f}")
    _debug_count += 1

def log_model_param_sums(model_manager: ModelReSwapManager):
    outputs = {}

    fn = lambda x: x.float().abs().sum().item()

    for name, module_manager in model_manager.module_managers.items():
        outputs[name] = module_manager.debug_maybe_swapped_flat_param_values(fn)

    logger.info(pformat(outputs))

def benchmark_multi_devices(
    mist_config: MistConfig,
):
    if mist_config.memory_debug:
        torch.cuda.memory._record_memory_history()

    # Reset mist patcher to disable the symbolic execution
    reset_mist_patcher()

    # Create the model config
    model_config = mist_config.model
    training_config = mist_config.training
    strategy_config = mist_config.strategy
    gradient_accumulation_steps = strategy_config.gradient_accumulation_steps
    global_batch_size = training_config.global_batch_size

    # Initialize the configs
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    num_nodes = world_size // torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Init the distributed process group
    dist.init_process_group(
        backend="nccl", timeout=timedelta(seconds=mist_config.nccl_timeout)
    )
    global_rank = dist.get_rank()
    # Init parallel state
    parallel_state.initialize_parallel(mist_config)
    num_pipeline_stages = parallel_state.get_num_pipeline_stages()
    stage_idx = parallel_state.get_pipeline_parallel_stage_idx()
    dp_process_group = parallel_state.get_data_parallel_group()
    dp_all_gather_process_group = parallel_state.get_data_parallel_all_gather_group()
    dp_reduce_scatter_process_group = (
        parallel_state.get_data_parallel_reduce_scatter_group()
    )
    tp_process_group = parallel_state.get_tensor_parallel_group()
    if tp_process_group is not None and tp_process_group.size() == 1:
        tp_process_group = None
    # Only print the logs in the first rank in the dp process group
    update_logger_formatter_for_rank(
        logger,
        disable_print=global_rank != parallel_state.get_data_parallel_global_ranks()[0],
    )

    # Extract the stage strategy
    stage_strategy = strategy_config.stage_strategies[stage_idx]
    per_device_batch_size = stage_strategy[0]
    dp_size = stage_strategy[1]
    tp_size = stage_strategy[2]
    shard_weights = stage_strategy[3]
    shard_grads = stage_strategy[4]
    shard_opts = stage_strategy[5]
    weight_swap_ratio = stage_strategy[6]
    grad_swap_ratio = stage_strategy[7]
    opt_swap_ratio = stage_strategy[8]
    activation_swap_ratio = stage_strategy[9]
    # Create sharding strategy
    sharding_strategy = HandleShardingStrategy.from_shard_flags(
        shard_weights, shard_grads, shard_opts
    )

    # Extract the pre- and post- strategy
    assert strategy_config.pre_post_strategy == "preset", (
        f"Only support preset pre-post strategy for now, "
        f"but got {strategy_config.pre_post_strategy}."
    )
    pre_post_strategy_array = strategy_config.pre_post_strategy_array
    pre_post_per_device_batch_size = pre_post_strategy_array[0]
    pre_post_dp_size = pre_post_strategy_array[1]
    pre_post_tp_size = pre_post_strategy_array[2]
    pre_post_shard_weights = pre_post_strategy_array[3]
    pre_post_shard_grads = pre_post_strategy_array[4]
    pre_post_shard_opts = pre_post_strategy_array[5]
    pre_post_weight_swap_ratio = pre_post_strategy_array[6]
    pre_post_grad_swap_ratio = pre_post_strategy_array[7]
    pre_post_opt_swap_ratio = pre_post_strategy_array[8]
    pre_post_activation_swap_ratio = pre_post_strategy_array[9]
    # Create pre-post sharding strategy
    pre_post_sharding_strategy = HandleShardingStrategy.from_shard_flags(
        pre_post_shard_weights, pre_post_shard_grads, pre_post_shard_opts
    )

    from torch.optim import AdamW

    # Get the extra configs
    optimizer_cls = FusedAdam
    optimizer_kwargs = {"lr": 1e-5, "betas": (0.9999, 0.9999)}
    optim_amp = training_config.optimizer_dtype != training_config.params_dtype
    scale = 2**12
    grad_scaler = ConstantGradScaler(scale=scale)

    # Model provider
    model_provider = partial(
        base_model_provider,
        model_name=model_config.name,
        model_config=model_config,
        num_hidden_layers=model_config.num_hidden_layers,
        pre_process=True,
        post_process=True,
    )

    # Inputs provider
    b = per_device_batch_size
    s = model_config.max_position_embeddings
    v = model_config.vocab_size

    base_inputs_provider = get_inputs_provider(model_config.name)
    inputs_provider = partial(
        base_inputs_provider,
        batch_size=b,
        seq_len=s,
        device=get_device(torch.cuda.current_device()),
    )

    # ==============================================================================
    # Pipe Module construction
    with init_empty_weights(enable=True, include_buffers=False):
        raw_model = model_provider(
            device="meta",
            process_groups=tp_process_group,
            pre_post_process_group=tp_process_group,
        )
    with init_empty_weights(enable=True, include_buffers=True):
        inputs = inputs_provider()

    graph, modules_to_graphs = mist_trace(
        raw_model,
        inputs,
        trace_into_submodule=False,
        device="meta",
        fallback_device=device,
    )
    pipe_modules: List[nn.Module] = build_pipe_modules_based_on_block_partition(
        model=raw_model,
        root_graph=graph,
        modules_to_graphs=modules_to_graphs,
        block_partition=strategy_config.layer_partitions,
        # inputs=inputs,    # If inputs are provided, then an extra forward pass will be performed to ensure the functionality
        raise_error_if_single_block=False,
    )
    model = pipe_modules[stage_idx]
    logger.info(f"Module Params: {count_module_parameters(model) / 1000 ** 3:.2f} B")
    logger.info(
        f"Memory Allocated Before Materialization: {torch.cuda.memory_allocated(device) / 1024 ** 2 : .2f} MB"
    )

    # ==============================================================================
    # Model Materialization
    # Process the block layers
    assert isinstance(
        model, PipeModule
    ), f"The following logic has only been tested for PipeModule, but got {type(model)}"
    for name, sub_module in model.named_modules():
        sub_module.name = name
    block_sub_modules: List[nn.Module] = get_default_sub_modules(
        model, return_names=False
    )
    # Gradient checkpointing modules
    num_ckpt_modules: int = strategy_config.gradient_checkpointing[stage_idx]
    # Construct the model re-swap manager
    sub_modules = block_sub_modules.copy()
    if model.pre_layer is not None:
        sub_modules.insert(0, model.pre_layer)
    if model.post_layer is not None:
        sub_modules.append(model.post_layer)
    assert len(sub_modules) > 1, "The model should have at least 2 sub-modules"
    # state swap ratios
    block_state_swap_ratio = (weight_swap_ratio, grad_swap_ratio)
    pre_post_state_swap_ratio = (
        pre_post_weight_swap_ratio,
        pre_post_grad_swap_ratio,
    )
    state_swap_ratios = {m.name: block_state_swap_ratio for m in block_sub_modules}
    activation_swap_ratios = {m.name: activation_swap_ratio for m in block_sub_modules}
    sharding_strategies = {m.name: sharding_strategy for m in block_sub_modules}
    process_groups = {m.name: dp_process_group for m in block_sub_modules}
    dp_all_gather_process_groups = {
        m.name: dp_all_gather_process_group for m in block_sub_modules
    }
    dp_reduce_scatter_process_groups = {
        m.name: dp_reduce_scatter_process_group for m in block_sub_modules
    }
    # ========================
    # TODO(zhanda): make it a configuration
    # sharding_strategies[block_sub_modules[0].name] = HandleShardingStrategy.OPT_ONLY
    # ========================
    opt_swap_ratios = {m.name: opt_swap_ratio for m in block_sub_modules}
    if model.pre_layer is not None:
        pre_layer_name = model.pre_layer.name
        state_swap_ratios[pre_layer_name] = pre_post_state_swap_ratio
        activation_swap_ratios[pre_layer_name] = 0.0
        sharding_strategies[pre_layer_name] = pre_post_sharding_strategy
        # TODO(zhanda): the process group for pre-layer and post-layer can be
        # different from that of the block layers
        process_groups[pre_layer_name] = dp_process_group
        dp_all_gather_process_groups[pre_layer_name] = dp_all_gather_process_group
        dp_reduce_scatter_process_groups[pre_layer_name] = (
            dp_reduce_scatter_process_group
        )
        opt_swap_ratios[pre_layer_name] = pre_post_opt_swap_ratio
    if model.post_layer is not None:
        post_layer_name = model.post_layer.name
        state_swap_ratios[post_layer_name] = pre_post_state_swap_ratio
        activation_swap_ratios[post_layer_name] = 0.0
        sharding_strategies[post_layer_name] = pre_post_sharding_strategy
        process_groups[post_layer_name] = dp_process_group
        dp_all_gather_process_groups[post_layer_name] = dp_all_gather_process_group
        dp_reduce_scatter_process_groups[post_layer_name] = (
            dp_reduce_scatter_process_group
        )
        opt_swap_ratios[post_layer_name] = pre_post_opt_swap_ratio

    # Create the model re-swap manager
    model_re_swap_manager = ModelReSwapManager(
        model=model,
        modules={m.name: m for m in sub_modules},
        module_sequence=[m.name for m in sub_modules],
        state_swap_ratios=state_swap_ratios,
        activation_swap_ratios=activation_swap_ratios,
        sharding_strategies=sharding_strategies,
        process_groups=process_groups,
        all_gather_process_groups=dp_all_gather_process_groups,
        reduce_scatter_process_groups=dp_reduce_scatter_process_groups,
        gradient_accumulation_steps=gradient_accumulation_steps,
        pipeline_stage_idx=stage_idx if strategy_config.num_stages > 1 else 1,
        num_pipeline_stages=strategy_config.num_stages,
        cuda_device=device,
        grad_scaler=grad_scaler,
        opt_swap_ratios=opt_swap_ratios,
    )

    for i, sub_module in enumerate(block_sub_modules):
        is_first = sub_module == sub_modules[0]
        is_last = sub_module == sub_modules[-1]
        materialize_module(sub_module, device=device, inplace=True)
        model_re_swap_manager.init_module(
            sub_module,
            # TODO(zhanda): design a heuristic to determine where to put the
            # activation checkpointing layers, e.g., the first and last few layers
            # activation_checkpointing=i < num_ckpt_modules,
            activation_checkpointing=i >= len(block_sub_modules) - num_ckpt_modules,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            is_first=is_first,
            is_last=is_last,
        )
        logger.info(f"Successfully initialized module {sub_module.name}.")
    for pre_post_layer in [model.pre_layer, model.post_layer]:
        if pre_post_layer is None:
            continue
        is_first = pre_post_layer == sub_modules[0]
        is_last = pre_post_layer == sub_modules[-1]
        materialize_module(pre_post_layer, device=device, inplace=True)
        model_re_swap_manager.init_module(
            pre_post_layer,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            is_first=is_first,
            is_last=is_last,
            use_post_accumulate_grad_hook=is_first,
        )
    # Check all the modules are initialized
    assert all_params_and_buffers_in_device(
        model, device, allow_cpu=True
    ), f"Not all the parameters and buffers are in the device {device}"
    # Create the buffers and register the hooks
    model_re_swap_manager.register_hooks()
    cuda_empty_cache()
    torch.cuda.synchronize()
    dist.barrier()

    # log_model_param_sums(model_re_swap_manager)

    # ==============================================================================

    # Benchmarking
    cuda_empty_cache()
    logger.info(
        f"Memory Allocated Before Benchmarking: {torch.cuda.memory_allocated(device) / 1024 ** 2 : .2f} MB"
    )
    torch.cuda.reset_peak_memory_stats()

    tiny_bench = (
        mist_config.profile or mist_config.memory_debug or mist_config.tiny_bench
    )
    if mist_config.profile:
        warmup = 0
        number = 1
    elif not tiny_bench:
        warmup = 5
        number = 5
    # elif world_size >= 8:
    else:
        warmup = 1
        # number = 10 is for correctness check while
        # number = 2 is for performance check
        number = 2

    # ==============================================================================

    def fwd_func(*unused_args, **unused_kwargs):
        if strategy_config.num_stages == 1 or parallel_state.is_pipeline_first_stage():
            inputs: Dict[str, Any] = inputs_provider()
            first_stage_module_signature = inspect.signature(model.forward)
            first_stage_module_bound_signature = first_stage_module_signature.bind(
                **inputs
            )
            model.input_tensors = list(
                first_stage_module_bound_signature.arguments.values()
            )
        assert model.input_tensors is not None
        inputs: List[torch.Tensor] = model.input_tensors
        loss = model(*inputs)
        if strategy_config.num_stages == 1 or parallel_state.is_pipeline_last_stage():
            if grad_scaler is not None:
                loss = grad_scaler.scale(loss)
        # logger.error(f"Loss: {loss.item()}")
        return loss

    if num_pipeline_stages > 1:

        def run_func():
            forward_backward_pipelining_without_interleaving(
                forward_step_func=fwd_func,
                data_iterator=None,
                model=model,
                model_re_swap_manager=model_re_swap_manager,
                config=mist_config,
                forward_only=False,
            )

        oom = torch.tensor([0], device=device)
        try:
            latencies, (
                peak_allocated_memories,
                peak_reserved_memories,
            ) = benchmark_func_cuda_event(
                run_func,
                warmup=warmup,
                number=number,
                prepare_func=None,
                sync_func=None,
                num_memory_records=10,
                enable_tqdm=False,
            )
        except RuntimeError as e:
            print(f"[RANK {dist.get_rank()}] Error: {e}")
            oom.fill_(1)
            find_and_kill_other_processes(
                pattern="benchmark_one_case.py",
                workers_for_pdsh=f"worker-[1-{num_nodes}]"
            )
            raise e

        # Check oom of all processes
        # torch.distributed.all_reduce(oom, op=torch.distributed.ReduceOp.SUM)

        torch.cuda.synchronize()
        dist.barrier()

        log_memory(
            peak_allocated_memories,
            peak_reserved_memories,
            "Pipeline Peak",
            print_fn=logger.info,
        )

        (
            latencies_mean,
            latencies_median,
            latencies_std,
        ) = process_benchmarking_results(
            latencies, msg="Pipeline", print_to_screen=True, print_fn=logger.info
        )

        # Format the results
        total_latency = latencies_mean

        # All-gather the results from different pipeline stages
        stage_peak_allocated_memories = torch.zeros(
            world_size, device=device, dtype=torch.float
        )
        stage_peak_allocated_memories[dist.get_rank()] = max(peak_allocated_memories)
        torch.distributed.all_reduce(stage_peak_allocated_memories)
        stage_peak_allocated_memories = (
            stage_peak_allocated_memories.view(num_pipeline_stages, -1)
            .max(dim=1)
            .values.tolist()
        )
        # Do the same for the reserved memories
        stage_peak_reserved_memories = torch.zeros(
            world_size, device=device, dtype=torch.float
        )
        stage_peak_reserved_memories[dist.get_rank()] = max(peak_reserved_memories)
        torch.distributed.all_reduce(stage_peak_reserved_memories)
        stage_peak_reserved_memories = (
            stage_peak_reserved_memories.view(num_pipeline_stages, -1)
            .max(dim=1)
            .values.tolist()
        )

    else:

        def run_func():
            for _ in range(gradient_accumulation_steps):
                loss = fwd_func()
                # torch.cuda.synchronize()
                loss.backward()
                # torch.cuda.synchronize()

        def sync_func():
            torch.cuda.synchronize()
            dist.barrier()

        oom = torch.tensor([0], device=device)
        try:
            latencies, (peak_allocated_memories, peak_reserved_memories) = (
                benchmark_func_walltime(
                    run_func,
                    warmup=warmup,
                    number=number,
                    prepare_func=None,
                    sync_func=sync_func,
                    num_memory_records=10,
                    enable_tqdm=False,
                )
            )
        except RuntimeError as e:
            print(f"[RANK {dist.get_rank()}] Error: {e}")
            oom.fill_(1)
            find_and_kill_other_processes(
                pattern="benchmark_one_case.py",
                workers_for_pdsh=f"worker-[1-{num_nodes}]"
            )
            raise e

        torch.cuda.synchronize()
        dist.barrier()

        mean, median, std = process_benchmarking_results(
            latencies, "Total", print_to_screen=True, print_fn=logger.info
        )

        # Format the results
        total_latency = mean
        stage_peak_allocated_memories = peak_allocated_memories
        stage_peak_reserved_memories = peak_reserved_memories

    # ==============================================================================
    # Profiling

    if getattr(mist_config._original_config, "profile", False):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs", worker_name=f"{time.strftime('%Y%m%d-%H%M%S')}-worker-{dist.get_rank()}-stage-{stage_idx}"),
            record_shapes=True,
            # with_stack=True,
            # with_modules=True,
        ) as profiler:
            for i in range(2):
                if num_pipeline_stages == 1:
                    for _ in range(gradient_accumulation_steps):
                        output = fwd_func()
                        loss = output.sum()
                        torch.cuda.synchronize()
                        loss.backward()
                        torch.cuda.synchronize()
                    torch.cuda.synchronize()

                else:
                    run_func()
                    torch.cuda.synchronize()
                    dist.barrier()

                # cuda_empty_cache()
                profiler.step()

    if getattr(mist_config._original_config, "memory_debug", False):
        torch.cuda.memory._dump_snapshot(f"memory_snapshot_{local_rank}.pickle")

    return total_latency, (stage_peak_allocated_memories, stage_peak_reserved_memories)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config.yaml",
)
def main(cfg: DictConfig) -> None:
    cfg.strategy.enabled = True
    cfg.tuning.enabled = False
    mist_config = MistConfig.from_dict_config(cfg)
    total_latency, (stage_peak_allocated_memories, stage_peak_reserved_memories) = (
        benchmark_multi_devices(mist_config)
    )
    logger.info(f"Total Latency: {total_latency:.4f}")
    log_memory(
        stage_peak_allocated_memories,
        stage_peak_reserved_memories,
        "Stage Peak",
        print_fn=logger.info,
    )

    # Output the total latency and peak memories if the output path is specified
    # Only for the main rank
    if dist.get_rank() == 0 and getattr(cfg, "output_path", None) is not None:
        folder_path = os.path.dirname(cfg.output_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        config_file_name = Path(os.path.basename(cfg.output_path)).stem
        # Save the total latency and peak memories to a summary file
        summary_path = os.path.join(folder_path, "summary.json")
        data = load_json(summary_path) if os.path.exists(summary_path) else {}
        key = config_file_name
        stage_peak_allocated_memories_str = ", ".join(
            [f"{mem:.0f} MB" for mem in stage_peak_allocated_memories]
        )
        stage_peak_allocated_memories_str = f"[{stage_peak_allocated_memories_str}]"
        stage_peak_reserved_memories_str = ", ".join(
            [f"{mem:.0f} MB" for mem in stage_peak_reserved_memories]
        )
        stage_peak_reserved_memories_str = f"[{stage_peak_reserved_memories_str}]"
        value = {
            "exec_total_cost": f"{total_latency:.4f}",
            "exec_stage_peak_allocated_memories": stage_peak_allocated_memories_str,
            "exec_stage_peak_reserved_memories": stage_peak_reserved_memories_str,
        }
        data.setdefault(key, {}).update(value)
        save_json(data, summary_path)


if __name__ == "__main__":
    main()
