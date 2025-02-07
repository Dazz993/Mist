import itertools
from numbers import Integral, Number
import os
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, cache
from itertools import product
from time import perf_counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional, Iterator, Any, Sequence, Callable

import numpy as np
import sympy as sp
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from more_itertools import chunked

from mist import global_symbol_manager as gsm
from mist.config import MistConfig
from mist.analyzer.info import LayerInfo
from mist.analyzer.strategy import (
    ModelStrategy,
    LayerStrategy,
    PhaseStrategy,
    create_model_strategy_from_layer_strategies,
)
from mist.analyzer.model_analyzer import ModelAnalyzer
from mist.analyzer.pipeline import (
    latency_for_pipe,
    latency_for_pipe_with_fixed_time_in_stage,
)
from mist.logger import get_logger
from mist.tools.optimize import (
    predict_gpu_gpu_comm_latency as raw_predict_gpu_gpu_comm_latency,
    predict_cpu_gpu_comm_latency as raw_predict_cpu_gpu_comm_latency,
    # gpu_to_gpu_only_latency_model,
    # cpu_to_gpu_only_latency_model,
    interference_estimate,
    interference_estimate_for_one_group,
)
from mist.tuning.optim_prob_base import OptimProb, _calculate_search_space_size
from mist.utils.pipeline_parallel import calculate_num_warmup_and_1f1b_phases
from mist.utils.sympy import (
    autowrap_with_cython,
    ufuncify_with_cython,
    lambdify_with_numpy,
)
from mist.utils.pareto import sample_pareto_frontier

# BATCH_SIZE_CANDIDATES = (1, 2, 3, 4, 8)
BATCH_SIZE_CANDIDATES = (1, 2, 4)
OFFLOADING_RATIO_CANDIDATES = (0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
SPARSE_OFFLOADING_RATIO_CANDIDATES = (0.0, 0.25, 0.375, 0.5, 0.75, 1.0)
# OFFLOADING_RATIO_CANDIDATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# OFFLOADING_RATIO_CANDIDATES = (0.0, 1.0)

MB = 1 << 20
GB = 1 << 30

logger = get_logger(__name__)

SYMPY_EVAL_FUNC = ufuncify_with_cython
MEM_INF = 1e12

SAMPLE_SIZE = 30


def _adjust_sympy_eval_func(samples: int):
    global SYMPY_EVAL_FUNC
    if samples > 50:
        SYMPY_EVAL_FUNC = ufuncify_with_cython
    else:
        SYMPY_EVAL_FUNC = lambdify_with_numpy


def np_product(a, b, combine_dims=False):
    """Calculate the product of two 2D numpy arrays.
    a: (N, P)
    b: (M, Q)
    result: (N * M, P + Q)
    """
    assert a.ndim == b.ndim == 2, f"a.ndim: {a.ndim}, b.ndim: {b.ndim}. a: {a}, b: {b}"
    # Repeat a for choice_size times and reshape
    a_repeated = np.repeat(a[:, np.newaxis, :], b.shape[0], axis=1)

    # Tile b for batch_size times
    b_tiled = np.tile(b[np.newaxis, :, :], (a.shape[0], 1, 1))

    # Combine a and b along the last axis
    result = np.concatenate([a_repeated, b_tiled], axis=-1)

    if combine_dims:
        return result.reshape(-1, result.shape[-1])
    else:
        return result


def get_concrete_mem_full_states(
    layer_info: LayerInfo, item: str, tp_size: Union[int, np.ndarray]
):
    assert item in ("weights", "grads", "opts")
    if not isinstance(tp_size, np.ndarray):
        tp_size = np.array([tp_size])
    tp_size = tp_size.astype(np.float64)
    mem_func = get_func_mem_full_states(layer_info, item=item)
    mem = mem_func(tp_size, 0)
    if isinstance(mem, np.ndarray) and mem.ndim == 0:
        mem = mem.reshape(1)
    elif isinstance(mem, Number):
        mem = np.array([mem])
    return mem


@cache
def get_func_mem_full_states(
    layer_info: LayerInfo,
    item: str,
):
    assert item in ("weights", "grads", "opts")
    mem_expr = getattr(layer_info.fwd_info, f"full_{item}")
    mem_expr = mem_expr / MB

    if not isinstance(mem_expr, sp.Expr):
        return lambda *args: mem_expr

    # V1: the orginal implementation
    # assert (
    #     len(mem_expr.free_symbols) == 1 and "tp" in list(mem_expr.free_symbols)[0].name
    # ), f"mem_expr: {mem_expr} with free_symbols: {mem_expr.free_symbols}. item: {item}"
    # tp_size_symbol = list(mem_expr.free_symbols)[0]
    # mem_func = SYMPY_EVAL_FUNC(args=(tp_size_symbol,), expr=mem_expr)

    # V2: to support create attn buffers on the fly
    strategy = layer_info.fwd_info.strategy
    symbols = (strategy.tp_size, strategy.per_device_batch_size)
    mem_func = SYMPY_EVAL_FUNC(args=symbols, expr=mem_expr)

    return mem_func


def get_concrete_mem_partial_states(
    full_states: np.ndarray,
    sharding_size: np.ndarray,
    offloading_ratio: np.ndarray,
):
    """
    Element-wise formula:
        partial_states = full_states / sharding_size * (1 - offloading_ratio)
    """
    partial_states = full_states / sharding_size * (1 - offloading_ratio)
    return partial_states


def get_concrete_mem_saved(
    layer_info: LayerInfo,
    ckpt: bool,
    batch_size_per_device: Union[int, np.ndarray],
    tp_size: Union[int, np.ndarray],
):
    if not isinstance(batch_size_per_device, np.ndarray):
        batch_size_per_device = np.array([batch_size_per_device])
    if not isinstance(tp_size, np.ndarray):
        tp_size = np.array([tp_size])
    batch_size_per_device = batch_size_per_device.astype(np.float64)
    tp_size = tp_size.astype(np.float64)
    mem_func = get_func_mem_saved(layer_info, ckpt=ckpt)
    return mem_func(batch_size_per_device, tp_size)


@cache
def get_func_mem_saved(
    layer_info: LayerInfo,
    ckpt: bool,
):
    strategy = layer_info.fwd_info.strategy
    symbols = (strategy.per_device_batch_size, strategy.tp_size)
    ckpt = 1.0 if ckpt else 0.0
    mem_expr = layer_info.fwd_info.saved.subs({strategy.ckpt: ckpt})
    mem_expr = mem_expr / MB
    mem_func = SYMPY_EVAL_FUNC(args=symbols, expr=mem_expr)
    return mem_func


def get_concrete_mem_peak(
    layer_info: LayerInfo,
    phase_type: str,
    batch_size_per_device: Union[int, np.ndarray],
    tp_size: Union[int, np.ndarray],
    ckpt: Optional[bool] = None,
):
    assert phase_type in ("fwd", "bwd"), f"phase_type: {phase_type}"
    if not isinstance(batch_size_per_device, np.ndarray):
        batch_size_per_device = np.array([batch_size_per_device])
    if not isinstance(tp_size, np.ndarray):
        tp_size = np.array([tp_size])
    batch_size_per_device = batch_size_per_device.astype(np.float64)
    tp_size = tp_size.astype(np.float64)
    mem_func = get_func_mem_peak(layer_info, phase_type=phase_type, ckpt=ckpt)
    return mem_func(batch_size_per_device, tp_size)


@cache
def get_func_mem_peak(
    layer_info: LayerInfo,
    phase_type: str,
    ckpt: bool,
):
    if phase_type == "fwd":
        strategy = layer_info.fwd_info.strategy
        symbols = (strategy.per_device_batch_size, strategy.tp_size)
        ckpt = 1.0 if ckpt else 0.0
        mem_expr = layer_info.fwd_info.peak.subs({strategy.ckpt: ckpt})
        mem_expr = mem_expr / MB
        mem_func = SYMPY_EVAL_FUNC(args=symbols, expr=mem_expr)
    elif phase_type == "bwd":
        strategy = layer_info.bwd_info.strategy
        symbols = (strategy.per_device_batch_size, strategy.tp_size)
        mem_expr = layer_info.bwd_info.peak
        mem_expr = mem_expr / MB
        mem_func = SYMPY_EVAL_FUNC(args=symbols, expr=mem_expr)
    else:
        raise ValueError(f"phase_type: {phase_type}")

    return mem_func


def get_concrete_mem_output(
    layer_info: LayerInfo,
    batch_size_per_device: Union[int, np.ndarray],
    tp_size: Union[int, np.ndarray],
):
    if not isinstance(batch_size_per_device, np.ndarray):
        batch_size_per_device = np.array([batch_size_per_device])
    if not isinstance(tp_size, np.ndarray):
        tp_size = np.array([tp_size])
    batch_size_per_device = batch_size_per_device.astype(np.float64)
    tp_size = tp_size.astype(np.float64)
    mem_func = get_func_mem_output(layer_info)
    return mem_func(batch_size_per_device, tp_size)


@cache
def get_func_mem_output(
    layer_info: LayerInfo,
):
    strategy = layer_info.fwd_info.strategy
    symbols = (strategy.per_device_batch_size, strategy.tp_size)
    mem_expr = layer_info.fwd_info.output
    mem_expr = mem_expr / MB
    if isinstance(mem_expr, Number):
        mem_func = lambda *args, **kwargs: mem_expr
    else:
        mem_func = SYMPY_EVAL_FUNC(args=symbols, expr=mem_expr)
    return mem_func


def get_concrete_latency_exec(
    layer_info: LayerInfo,
    batch_size_per_device: np.ndarray,
    tp_size: np.ndarray,
    inter_size: np.ndarray,
    intra_size: np.ndarray,
    gpu_gpu_comm_params: Tuple[float, float, float],
    gradient_accumulation_steps: int,
):
    """Get the concrete latency of the layer.

    Parameters
    ----------
    layer_info
        layer_info object
    batch_size_per_device
        the ndarray of batch_size_per_device
    tp_size
        the ndarray of tp_size
    inter_size
        the communication that happens in inter-node
    intra_size
        the communication that happens in intra-node
    gpu_gpu_comm_params
        the communication parameters
    gradient_accumulation_steps
        the number of gradient accumulation steps
    """
    # Get unique batch_size_per_device and tp_size
    # Concatenate batch_size_per_device and tp_size / Stack
    if not isinstance(batch_size_per_device, np.ndarray):
        batch_size_per_device = np.array([batch_size_per_device])
    if not isinstance(tp_size, np.ndarray):
        tp_size = np.array([tp_size])
    assert batch_size_per_device.shape == tp_size.shape

    latencies = np.array(
        [
            _single_get_concrete_latency_exec(
                layer_info,
                batch_size_per_device[i],
                tp_size[i],
                (
                    inter_size[i]
                    if isinstance(inter_size, np.ndarray) and inter_size.ndim > 0
                    else inter_size
                ),
                (
                    intra_size[i]
                    if isinstance(intra_size, np.ndarray) and inter_size.ndim > 0
                    else intra_size
                ),
                gpu_gpu_comm_params,
                gradient_accumulation_steps,
            )
            for i in range(len(batch_size_per_device))
        ]
    )
    fwd_latencies = latencies[:, 0]
    bwd_latencies = latencies[:, 1]
    extra_fwd_latencies = latencies[:, 2]
    extra_bwd_latencies = latencies[:, 3]
    return fwd_latencies, bwd_latencies, extra_fwd_latencies, extra_bwd_latencies


@cache
def _single_get_concrete_latency_exec(
    layer_info: LayerInfo,
    batch_size_per_device: float,
    tp_size: float,
    inter_size: np.ndarray,
    intra_size: np.ndarray,
    gpu_gpu_comm_params: Tuple[float, float, float],
    gradient_accumulation_steps: int,
):
    factory_kwargs = {
        "inter_size": inter_size,
        "intra_size": intra_size,
        "gpu_gpu_comm_params": gpu_gpu_comm_params,
    }
    fwd_strategy = layer_info.strategy.fwd_strategy
    symbol_mapping = {
        fwd_strategy.per_device_batch_size: int(batch_size_per_device),
        fwd_strategy.tp_size: int(tp_size),
    }
    fwd_latencies = 0
    bwd_latencies = 0
    extra_fwd_latencies = 0
    extra_bwd_latencies = 0
    fwd_latencies_list = []
    bwd_latencies_list = []
    extra_fwd_latencies_list = []
    extra_bwd_latencies_list = []
    latencies = []
    for symbolic_node_spec in layer_info.fwd_info._symbolic_node_specs:
        concrete_node_spec = symbolic_node_spec.concretize(symbol_mapping)
        logger.debug(
            f"Begin to profile: \n"
            f"\t[Sym] {symbolic_node_spec}, \n"
            f"\t[Con] {concrete_node_spec}, \n"
            f"\t[SymMap] {symbol_mapping}"
        )
        (
            (fwd_latency, _, _),
            (bwd_latency, _, _),
            extra_fwd_latency,
            extra_bwd_latency,
        ) = concrete_node_spec.profile(**factory_kwargs)

        # TODO(zhanda): fix the backward latency issue
        if bwd_latency != 0 and bwd_latency > 2.3 * fwd_latency:
            bwd_latency = fwd_latency * 2
        fwd_latency += extra_fwd_latency
        bwd_latency += extra_bwd_latency

        if fwd_latency < 5e-5 or bwd_latency < 1e-4:
            continue

        logger.debug(
            f"\n\tconcrete_node_spec: {concrete_node_spec}: \n"
            f"\t\t - [fwd_latency] {fwd_latency * 1e3:.4f} ms, "
            f"[extra_fwd_latency]: {extra_fwd_latency * 1e3:.4f} ms, \n"
            f"\t\t - [bwd_latency] {bwd_latency * 1e3:.4f} ms, "
            f"[extra_bwd_latency]: {extra_bwd_latency * 1e3:.4f} ms"
        )

        fwd_latencies += fwd_latency
        bwd_latencies += bwd_latency
        extra_fwd_latencies += extra_fwd_latency
        extra_bwd_latencies += extra_bwd_latency
        fwd_latencies_list.append(fwd_latency)
        bwd_latencies_list.append(bwd_latency)
        extra_fwd_latencies_list.append(extra_fwd_latency)
        extra_bwd_latencies_list.append(extra_bwd_latency)
        latencies.append(
            (
                concrete_node_spec,
                fwd_latency,
                bwd_latency,
                extra_fwd_latency,
                extra_bwd_latency,
            )
        )

    for symbolic_node_spec in layer_info.bwd_info._grad_accumulation_node_specs:
        concrete_node_spec = symbolic_node_spec.concretize(symbol_mapping)
        (grad_accumu_latency, _, _), (_, _, _), _, _ = concrete_node_spec.profile(
            **factory_kwargs
        )
        bwd_latencies += (
            grad_accumu_latency
            * (gradient_accumulation_steps - 1)
            / gradient_accumulation_steps
        )

    return fwd_latencies, bwd_latencies, extra_fwd_latencies, extra_bwd_latencies


@cache
def create_search_space_for_a_block_layer(
    num_nodes: int,
    num_gpus_per_node: int,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    batch_size_candidates: Tuple[int] = BATCH_SIZE_CANDIDATES,
    offloading_ratio_candidates: Tuple[float] = OFFLOADING_RATIO_CANDIDATES,
    sparse_offloading_ratio_candidates: Tuple[
        float
    ] = SPARSE_OFFLOADING_RATIO_CANDIDATES,
    constraint_fn: Optional[Callable] = None,
    tp_size: Optional[int] = None,
    fixed_ao_ratio: Optional[float] = None,
    fixed_oo_ratio: Optional[float] = None,
    fixed_go_ratio: Optional[float] = None,
    fixed_wo_ratio: Optional[float] = None,
    config: Optional[MistConfig] = None,
):
    num_gpus = num_nodes * num_gpus_per_node
    state_offloading_enabled = False
    activation_offloading_enabled = False
    if config is not None:
        state_offloading_enabled = config.tuning.state_offloading_enabled
        activation_offloading_enabled = config.tuning.activation_offloading_enabled

    batch_size_per_micro_batch = global_batch_size // gradient_accumulation_steps

    # * Parallelism
    parallelism_candidates = []
    for batch_size in batch_size_candidates:
        # DP Size is determined by local batch size and global batch size
        if (
            batch_size > batch_size_per_micro_batch
            or batch_size_per_micro_batch % batch_size != 0
        ):
            continue
        dp_size = batch_size_per_micro_batch // batch_size

        # TP size is determined by DP size and number of GPUs
        if dp_size > num_gpus or num_gpus % dp_size != 0:
            continue
        curr_tp_size = num_gpus // dp_size

        if constraint_fn is not None:
            if not constraint_fn(batch_size, curr_tp_size):
                continue

        tp_size = curr_tp_size if tp_size is None else tp_size
        if tp_size != curr_tp_size:
            continue

        # Heuristic 1: TP size
        if tp_size > num_gpus_per_node:
            continue
        if getattr(config, "disable_tp_tuning", False):
            if tp_size > 1:
                continue

        # Heuristic 2: No need to zero if DP == 1
        if dp_size == 1 or not config.tuning.zero_2_and_3_enabled:
            redundancy_sharding_flag_choices = [(0, 0, 1)]
        else:
            redundancy_sharding_flag_choices = [
                # (0, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (1, 1, 1),
            ]
        for ws_size, gs_size, oo_size in redundancy_sharding_flag_choices:
            parallelism_candidates.append(
                (batch_size, dp_size, tp_size, ws_size, gs_size, oo_size)
            )
    parallelism_np = np.array(parallelism_candidates).reshape(-1, 6)

    # * Offloading
    if state_offloading_enabled:
        wo_ratio_choices = sparse_offloading_ratio_candidates
        go_ratio_choices = sparse_offloading_ratio_candidates
        oo_ratio_choices = offloading_ratio_candidates
        # Used for tuning time benchmark
        if getattr(config, "disable_wo_tuning", False):
            wo_ratio_choices = [0.0]
        if getattr(config, "disable_go_tuning", False):
            go_ratio_choices = [0.0]
        if getattr(config, "disable_oo_tuning", False):
            oo_ratio_choices = [0.0]
    else:
        wo_ratio_choices = [0.0]
        go_ratio_choices = [0.0]
        oo_ratio_choices = [0.0]

    if fixed_wo_ratio is not None:
        wo_ratio_choices = [fixed_wo_ratio]
    if fixed_go_ratio is not None:
        go_ratio_choices = [fixed_go_ratio]
    if fixed_oo_ratio is not None:
        oo_ratio_choices = [fixed_oo_ratio]
        
    if activation_offloading_enabled:
        ao_ratio_choices = offloading_ratio_candidates
    elif fixed_ao_ratio is not None:
        ao_ratio_choices = [fixed_ao_ratio]
    else:
        ao_ratio_choices = [0.0]
    offloading_ratio_candidates = list(
        product(wo_ratio_choices, go_ratio_choices, oo_ratio_choices, ao_ratio_choices)
    )
    # Remove candidates that wo_ratio > 0 and go_ratio == 0
    offloading_ratio_candidates = [
        offloading_ratio
        for offloading_ratio in offloading_ratio_candidates
        if not (offloading_ratio[0] > 0 and offloading_ratio[1] == 0)
    ]
    offloading_ratio_np = np.array(offloading_ratio_candidates)

    # Combine parallelism and offloading
    search_space = np_product(parallelism_np, offloading_ratio_np, combine_dims=True)

    return search_space


@cache
def get_func_constraints_semantic(layer_info: LayerInfo):
    strategy = layer_info.fwd_info.strategy
    # ================================================================================
    # Deprecated because utils/autowrap has already done the float conversion
    # # Because batch size and tp size are created as integers,
    # # when codegen with cython, it will be converted to C int type.
    # # And C int type division will be floor division, which is not what we want.
    # float_batch_size, float_tp_size = sp.symbols("_float_batch_size, _float_tp_size")
    # symbols = (float_batch_size, float_tp_size)
    # mapping = {
    #     strategy.per_device_batch_size: float_batch_size,
    #     strategy.tp_size: float_tp_size,
    # }
    # constraint_exprs = [expr.subs(mapping) for expr in layer_info.constraints]
    # funcs = [autowrap_with_cython(args=symbols, expr=expr) for expr in constraint_exprs]
    # ================================================================================
    # Constraint func is actually not a heavy bottleneck, so we can use lambdify instead
    # of autowrap
    symbols = (strategy.per_device_batch_size, strategy.tp_size)
    funcs = [
        lambdify_with_numpy(args=symbols, expr=expr) for expr in layer_info.constraints
    ]

    def is_integer(num):
        return isinstance(num, Integral) or num == int(num)

    def fn(batch_size_per_device, tp_size):
        valid = all(is_integer(func(batch_size_per_device, tp_size)) for func in funcs)
        if not valid:
            # # For debugging
            # logger.debug(
            #     f"batch_size_per_device: {batch_size_per_device}, tp_size: {tp_size} is not valid. "
            #     f"constraint_exprs: {constraint_exprs}. "
            #     f"concrete values: {[func(batch_size_per_device, tp_size) for func in funcs]}"
            # )
            pass
        return valid

    return fn


def batched_tune_best_latency_for_stage(
    block_layer_info: LayerInfo,
    pre_layer_info: LayerInfo,
    post_layer_info: LayerInfo,
    pre_saved_micro_batches_candidates: Sequence[int],
    num_layers_candidates: Sequence[int],
    num_ckpt_layers_candidates: Union[Sequence[int], str],
    num_nodes: int,
    num_gpus_per_node: int,
    gradient_accumulation_steps: int,
    config: MistConfig,
    tp_size: Optional[int] = None,
    sample_size: int = None,
):
    # TODO(zhanda): Support non sequential candidates
    assert set(pre_saved_micro_batches_candidates) == set(
        range(0, max(pre_saved_micro_batches_candidates) + 1)
    ), f"pre_saved_micro_batches_candidates: {pre_saved_micro_batches_candidates} is not sequential"
    assert set(num_layers_candidates) == set(
        range(1, max(num_layers_candidates) + 1)
    ), f"num_layers_candidates: {num_layers_candidates} is not sequential"
    assert set(num_ckpt_layers_candidates) == set(
        range(0, max(num_ckpt_layers_candidates) + 1)
    ), f"num_ckpt_layers_candidates: {num_ckpt_layers_candidates} is not sequential"

    # Set sample size
    if sample_size is None:
        sample_size = config.sample_size

    # Create search space and combine with features_np
    # * parallelism (batch_size, dp_size, tp_size, wre_enabled, gre_enabled, ore_enabled)
    #                ^--- 0,     ^--- 1,  ^--- 2,  ^--- 3,      ^--- 4,      ^--- 5,
    # * offloading (wo_ratio, go_ratio, oo_ratio, ac_ratio)
    #               ^--- 6,   ^--- 7,   ^--- 8,   ^--- 9
    block_layer_constraints_fn = get_func_constraints_semantic(block_layer_info)
    search_space = create_search_space_for_a_block_layer(
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        global_batch_size=config.training.global_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        constraint_fn=block_layer_constraints_fn,
        tp_size=tp_size,
        config=config,
        fixed_wo_ratio=getattr(config, "fixed_wo_ratio", None),
        fixed_go_ratio=getattr(config, "fixed_go_ratio", None),
        fixed_oo_ratio=getattr(config, "fixed_oo_ratio", None),
        fixed_ao_ratio=getattr(config, "fixed_ao_ratio", None),
    )

    # If no candidates found in search space found, return None
    p = max(pre_saved_micro_batches_candidates)
    l = max(num_layers_candidates)
    c = max(num_ckpt_layers_candidates)
    s = sample_size
    f = search_space.shape[1]
    default_costs_stable = np.full((p + 1, l, c + 1, s), np.inf)
    default_costs_stable_with_pre = np.full((p + 1, l, c + 1, s), np.inf)
    default_costs_stable_with_post = np.full((l, c + 1, s), np.inf)
    default_costs_stable_no_pp = np.full((c + 1, s), np.inf)
    default_costs_delta = np.full((p + 1, l, c + 1, s), np.inf)
    default_costs_delta_with_pre = np.full((p + 1, l, c + 1, s), np.inf)
    default_costs_delta_with_post = np.full((l, c + 1, s), np.inf)
    default_costs_delta_no_pp = np.full((c + 1, s), np.inf)
    default_solutions = np.full((p + 1, l, c + 1, s, f), np.inf)
    default_solutions_with_pre = np.full((p + 1, l, c + 1, s, f), np.inf)
    default_solutions_with_post = np.full((l, c + 1, s, f), np.inf)
    default_solutions_no_pp = np.full((c + 1, s, f), np.inf)
    if search_space.size == 0:
        return (
            (default_costs_stable, default_costs_delta, default_solutions),
            (
                default_costs_stable_with_pre,
                default_costs_delta_with_pre,
                default_solutions_with_pre,
            ),
            (
                default_costs_stable_with_post,
                default_costs_delta_with_post,
                default_solutions_with_post,
            ),
            (
                default_costs_stable_no_pp,
                default_costs_delta_no_pp,
                default_solutions_no_pp,
            ),
        )

    results = batched_stage_analyze(
        block_layer_info=block_layer_info,
        pre_layer_info=pre_layer_info,
        post_layer_info=post_layer_info,
        pre_saved_micro_batches_candidates=pre_saved_micro_batches_candidates,
        num_layers_candidates=num_layers_candidates,
        num_ckpt_layers_candidates=num_ckpt_layers_candidates,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        gradient_accumulation_steps=gradient_accumulation_steps,
        stage_strategies=search_space,
        config=config,
        get_best_solution=True,
        sample_size=sample_size,
    )

    return results


def batched_stage_analyze(
    block_layer_info: LayerInfo,
    pre_layer_info: LayerInfo,
    post_layer_info: LayerInfo,
    pre_saved_micro_batches_candidates: Sequence[int],
    num_layers_candidates: Sequence[int],
    num_ckpt_layers_candidates: Union[Sequence[int], str],
    num_nodes: int,
    num_gpus_per_node: int,
    gradient_accumulation_steps: int,
    stage_strategies: List[List[float]],
    config: MistConfig,
    sample_size: int = SAMPLE_SIZE,
    get_best_solution: bool = False,
):
    # * parallelism (batch_size, dp_size, tp_size, wre_enabled, gre_enabled, ore_enabled)
    #                ^--- 0,     ^--- 1,  ^--- 2,  ^--- 3,      ^--- 4,      ^--- 5,
    # * offloading (wo_ratio, go_ratio, oo_ratio, ac_ratio)
    #               ^--- 6,   ^--- 7,   ^--- 8,   ^--- 9
    if isinstance(stage_strategies, (tuple, list)):
        stage_strategies = np.array(stage_strategies).reshape(-1, 10)

    if stage_strategies.size == 0:
        raise ValueError("stage_strategies is empty")

    # Change the sympy eval func according to the number of samples
    # this is a trade-off between the compilation time and computation time
    _adjust_sympy_eval_func(len(stage_strategies))

    if not get_best_solution:
        # Keep all the solutions and analyzed results, so we output four dicts
        # which maps (p, l, c)/(l, c)/(c,) to three np.ndarray (latency, mem_fwd_peak, mem_bwd_peak)
        results: Dict[Tuple[int, int, int], Tuple[Any]] = {}
        results_with_pre: Dict[Tuple[int, int], Tuple[Any]] = {}
        results_with_post: Dict[Tuple[int, int, int], Tuple[Any]] = {}
        reulst_with_pre_and_post: Dict[Tuple[int], Tuple[Any]] = {}
    else:
        # If we only need the best solution, we only keep the best solution and the corresponding
        # analyzed results, so we output the best latency for each (p, l, c)/(l, c)/(c,)
        # and pack them into nd.array
        p = max(pre_saved_micro_batches_candidates)
        l = max(num_layers_candidates)
        c = max(num_ckpt_layers_candidates)
        s = sample_size
        f = stage_strategies.shape[1]
        costs_stable = np.full((p + 1, l, c + 1, s), np.inf)
        costs_stable_with_pre = np.full((p + 1, l, c + 1, s), np.inf)
        costs_stable_with_post = np.full((l, c + 1, s), np.inf)
        costs_stable_no_pp = np.full((c + 1, s), np.inf)
        costs_delta = np.full((p + 1, l, c + 1, s), np.inf)
        costs_delta_with_pre = np.full((p + 1, l, c + 1, s), np.inf)
        costs_delta_with_post = np.full((l, c + 1, s), np.inf)
        costs_delta_no_pp = np.full((c + 1, s), np.inf)
        solutions = np.full((p + 1, l, c + 1, s, f), np.inf)
        solutions_with_pre = np.full((p + 1, l, c + 1, s, f), np.inf)
        solutions_with_post = np.full((l, c + 1, s, f), np.inf)
        solutions_no_pp = np.full((c + 1, s, f), np.inf)

    interference_model_params = config.hardware.interference_model_params
    gpu_gpu_bw_kwargs = {
        "gpu_gpu_comm_params": config.hardware.gpu_gpu_comm_params,
    }
    enable_advanced_opt_in_first_block = config.enable_advanced_opt_in_first_block

    def predict_gpu_gpu_comm_latency(
        op_name: str,
        mbytes: Union[int, np.ndarray],
        inter_size: Union[int, np.ndarray],
        intra_size: Union[int, np.ndarray],
    ):
        return raw_predict_gpu_gpu_comm_latency(
            op_name=op_name,
            gbytes=mbytes / 1024,
            inter_size=inter_size,
            intra_size=intra_size,
            **gpu_gpu_bw_kwargs,
        )

    def predict_cpu_gpu_comm_latency(mbytes: Union[int, np.ndarray]):
        return raw_predict_cpu_gpu_comm_latency(
            gbytes=mbytes / 1024,
            params=config.hardware.cpu_gpu_comm_params,
        )

    def predict_gpu_cpu_comm_latency(mbytes: Union[int, np.ndarray]):
        return raw_predict_cpu_gpu_comm_latency(
            gbytes=mbytes / 1024,
            params=config.hardware.gpu_cpu_comm_params,
        )

    # Get the information of the pre and post layers
    pre_post_strategy = None
    if not get_best_solution:
        assert config.strategy is not None
        assert config.strategy.pre_post_strategy is not None
        pre_post_strategy = config.strategy.pre_post_strategy
    else:
        assert config.tuning is not None
        assert config.tuning.pre_post_strategy is not None
        pre_post_strategy = config.tuning.pre_post_strategy

    if pre_post_strategy == "tuned":
        # We only tune the parallelism schedules for the pre and post layers
        pre_post_dp_size = stage_strategies[:, 1]
        pre_post_tp_size = stage_strategies[:, 2]
        pre_post_wre_enabled = stage_strategies[:, 3]
        pre_post_gre_enabled = stage_strategies[:, 4]
        pre_post_ore_enabled = stage_strategies[:, 5]
        pre_post_wo_ratio = 0.0
        pre_post_go_ratio = 0.0
        pre_post_oo_ratio = 0.0
        pre_post_ac_ratio = 0.0

    elif pre_post_strategy == "preset":
        pre_post_dp_size = config.strategy.pre_post_strategy_array[1]
        pre_post_tp_size = config.strategy.pre_post_strategy_array[2]
        pre_post_wre_enabled = config.strategy.pre_post_strategy_array[3]
        pre_post_gre_enabled = config.strategy.pre_post_strategy_array[4]
        pre_post_ore_enabled = config.strategy.pre_post_strategy_array[5]
        pre_post_wo_ratio = config.strategy.pre_post_strategy_array[6]
        pre_post_go_ratio = config.strategy.pre_post_strategy_array[7]
        pre_post_oo_ratio = config.strategy.pre_post_strategy_array[8]
        pre_post_ac_ratio = config.strategy.pre_post_strategy_array[9]

    elif pre_post_strategy == "dp":
        pre_post_dp_size = num_nodes * num_gpus_per_node
        pre_post_tp_size = 1
        pre_post_wre_enabled = False
        pre_post_gre_enabled = False
        pre_post_ore_enabled = True
        pre_post_wo_ratio = 0.0
        pre_post_go_ratio = 0.0
        pre_post_oo_ratio = 0.0
        pre_post_ac_ratio = 0.0

    elif pre_post_strategy.startswith("intra-node-tp"):
        assert pre_post_strategy in (
            "intra-node-tp-with-ore",
            "intra-node-tp-without-ore",
        )
        # for the pre- and post- layers, we assume that the tp size is `num_gpus_per_node`
        # and the opt redundancy sharding is `num_nodes`
        pre_post_dp_size = num_nodes
        pre_post_tp_size = num_gpus_per_node
        pre_post_wre_enabled = False
        pre_post_gre_enabled = True
        if pre_post_strategy == "intra-node-tp-with-ore":
            pre_post_ore_enabled = True
        else:
            pre_post_ore_enabled = False
        pre_post_wo_ratio = 0.0
        pre_post_go_ratio = 0.0
        pre_post_oo_ratio = 0.0
        pre_post_ac_ratio = 0.0

    else:
        raise NotImplementedError(
            f"config.strategy.pre_post_strategy: {config.strategy.pre_post_strategy}"
        )

    # Convert to np.ndarray
    def _convert_to_np_array(x):
        if isinstance(x, np.ndarray) and x.ndim > 0:
            return x
        else:
            return np.array([x])

    pre_post_dp_size = _convert_to_np_array(pre_post_dp_size)
    pre_post_tp_size = _convert_to_np_array(pre_post_tp_size)
    pre_post_wre_enabled = _convert_to_np_array(pre_post_wre_enabled)
    pre_post_gre_enabled = _convert_to_np_array(pre_post_gre_enabled)
    pre_post_ore_enabled = _convert_to_np_array(pre_post_ore_enabled)
    pre_post_wo_ratio = _convert_to_np_array(pre_post_wo_ratio)
    pre_post_go_ratio = _convert_to_np_array(pre_post_go_ratio)
    pre_post_oo_ratio = _convert_to_np_array(pre_post_oo_ratio)
    pre_post_ac_ratio = _convert_to_np_array(pre_post_ac_ratio)

    pre_post_batch_size_per_device = (
        config.training.global_batch_size
        // gradient_accumulation_steps
        // pre_post_dp_size
    )

    if isinstance(pre_post_batch_size_per_device, Number):
        if pre_post_batch_size_per_device < 1 or pre_post_batch_size_per_device > max(
            BATCH_SIZE_CANDIDATES
        ):
            if not get_best_solution:
                return (
                    results,
                    results_with_pre,
                    results_with_post,
                    reulst_with_pre_and_post,
                )
            else:
                return (
                    (costs_stable, costs_delta, solutions),
                    (costs_stable_with_pre, costs_delta_with_pre, solutions_with_pre),
                    (
                        costs_stable_with_post,
                        costs_delta_with_post,
                        solutions_with_post,
                    ),
                    (costs_stable_no_pp, costs_delta_no_pp, solutions_no_pp),
                )

        assert (
            pre_post_batch_size_per_device
            * pre_post_dp_size
            * gradient_accumulation_steps
            == config.training.global_batch_size
        ), (
            f"pre_post_batch_size_per_device: {pre_post_batch_size_per_device}, "
            f"pre_post_dp_size: {pre_post_dp_size}, "
            f"gradient_accumulation_steps: {gradient_accumulation_steps}, "
            f"config.global_batch_size: {config.training.global_batch_size}"
        )
        assert pre_post_dp_size * pre_post_tp_size == num_nodes * num_gpus_per_node, (
            f"pre_post_dp_size: {pre_post_dp_size}, "
            f"pre_post_tp_size: {pre_post_tp_size}, "
            f"num_nodes: {num_nodes}, "
            f"num_gpus_per_node: {num_gpus_per_node}"
        )
        # Not supported cases
        assert (
            pre_post_ac_ratio == 0.0
        ), f"pre_post_ac_ratio: {pre_post_ac_ratio}. Not Supported Net"
        assert pre_post_wre_enabled == pre_post_gre_enabled == False, (
            f"pre_post_wre_enabled: {pre_post_wre_enabled}, pre_post_gre_enabled: {pre_post_gre_enabled}. "
            "Not Supported Net"
        )

    # mem states and saved
    mem_pre_layer_full_weights = get_concrete_mem_full_states(
        pre_layer_info,
        item="weights",
        tp_size=pre_post_tp_size,
    )
    mem_pre_layer_full_grads = get_concrete_mem_full_states(
        pre_layer_info,
        item="grads",
        tp_size=pre_post_tp_size,
    )
    mem_pre_layer_full_opts = get_concrete_mem_full_states(
        pre_layer_info,
        item="opts",
        tp_size=pre_post_tp_size,
    )
    mem_pre_layer_partial_weights = get_concrete_mem_partial_states(
        mem_pre_layer_full_weights,
        sharding_size=np.where(pre_post_wre_enabled, pre_post_dp_size, 1),
        offloading_ratio=pre_post_wo_ratio,
    )
    mem_pre_layer_partial_grads = get_concrete_mem_partial_states(
        mem_pre_layer_full_grads,
        sharding_size=np.where(pre_post_gre_enabled, pre_post_dp_size, 1),
        offloading_ratio=pre_post_go_ratio,
    )
    mem_pre_layer_partial_opts = get_concrete_mem_partial_states(
        mem_pre_layer_full_opts,
        sharding_size=np.where(pre_post_ore_enabled, pre_post_dp_size, 1),
        offloading_ratio=pre_post_oo_ratio,
    )
    mem_pre_layer_states = (
        mem_pre_layer_partial_weights
        + mem_pre_layer_partial_grads
        + mem_pre_layer_partial_opts
    )
    mem_pre_layer_saved = get_concrete_mem_saved(
        pre_layer_info,
        ckpt=False,
        batch_size_per_device=pre_post_batch_size_per_device,
        tp_size=pre_post_tp_size,
    )
    mem_pre_layer_states_and_saved = mem_pre_layer_states + mem_pre_layer_saved
    mem_post_layer_full_weights = get_concrete_mem_full_states(
        post_layer_info,
        item="weights",
        tp_size=pre_post_tp_size,
    )
    mem_post_layer_full_grads = get_concrete_mem_full_states(
        post_layer_info,
        item="grads",
        tp_size=pre_post_tp_size,
    )
    mem_post_layer_full_opts = get_concrete_mem_full_states(
        post_layer_info,
        item="opts",
        tp_size=pre_post_tp_size,
    )
    mem_post_layer_partial_opts = get_concrete_mem_partial_states(
        mem_post_layer_full_opts,
        sharding_size=np.where(pre_post_ore_enabled, pre_post_dp_size, 1),
        offloading_ratio=pre_post_oo_ratio,
    )
    # Post layer's states are not offloaded
    # because when we calculate the peak memory, it happens either the last layer of the blocks
    # or the post layer
    mem_post_layer_states = (
        mem_post_layer_full_weights
        + mem_post_layer_full_grads
        + mem_post_layer_partial_opts
    )
    mem_post_layer_saved = get_concrete_mem_saved(
        post_layer_info,
        ckpt=False,
        batch_size_per_device=pre_post_batch_size_per_device,
        tp_size=pre_post_tp_size,
    )

    # mem peaks
    mem_post_layer_fwd_peak = get_concrete_mem_peak(
        post_layer_info,
        phase_type="fwd",
        batch_size_per_device=pre_post_batch_size_per_device,
        tp_size=pre_post_tp_size,
    )
    mem_post_layer_bwd_peak = get_concrete_mem_peak(
        post_layer_info,
        phase_type="bwd",
        batch_size_per_device=pre_post_batch_size_per_device,
        tp_size=pre_post_tp_size,
    )

    # mem output
    mem_post_layer_output = get_concrete_mem_output(
        post_layer_info,
        batch_size_per_device=pre_post_batch_size_per_device,
        tp_size=pre_post_tp_size,
    )

    # latencies
    pre_fwd_latency, pre_bwd_latency, pre_fwd_extra_latency, pre_bwd_extra_latency = (
        get_concrete_latency_exec(
            pre_layer_info,
            batch_size_per_device=pre_post_batch_size_per_device,
            tp_size=pre_post_tp_size,
            inter_size=np.where(
                pre_post_tp_size > num_gpus_per_node,
                pre_post_tp_size // num_gpus_per_node,
                1,
            ),
            intra_size=np.where(
                pre_post_tp_size > num_gpus_per_node,
                num_gpus_per_node,
                pre_post_tp_size,
            ),
            gradient_accumulation_steps=gradient_accumulation_steps,
            **gpu_gpu_bw_kwargs,
        )
    )
    (
        post_fwd_latency,
        post_bwd_latency,
        post_fwd_extra_latency,
        post_bwd_extra_latency,
    ) = get_concrete_latency_exec(
        post_layer_info,
        batch_size_per_device=pre_post_batch_size_per_device,
        tp_size=pre_post_tp_size,
        inter_size=np.where(
            pre_post_tp_size > num_gpus_per_node,
            pre_post_tp_size // num_gpus_per_node,
            1,
        ),
        intra_size=np.where(
            pre_post_tp_size > num_gpus_per_node, num_gpus_per_node, pre_post_tp_size
        ),
        gradient_accumulation_steps=gradient_accumulation_steps,
        **gpu_gpu_bw_kwargs,
    )
    # Get the pre and post layer's grad sync latency
    _pre_post_dp_comm_kwargs = {
        "inter_size": np.where(
            pre_post_dp_size > num_nodes, num_nodes, pre_post_dp_size
        ),
        "intra_size": np.where(
            pre_post_dp_size > num_nodes, pre_post_dp_size // num_nodes, 1
        ),
    }
    # Pre post layer weights all gathering
    # This weights all gather latency is only possible for the sharded opts
    pre_weights_all_gather_latency = predict_gpu_gpu_comm_latency(
        op_name="all_gather",
        mbytes=mem_pre_layer_full_weights,
        **_pre_post_dp_comm_kwargs,
    )
    pre_weights_all_gather_latency = np.where(
        pre_post_dp_size == 1, 0, pre_weights_all_gather_latency
    )
    pre_weights_all_gather_latency_amortized = np.where(
        pre_post_wre_enabled == 1,
        pre_weights_all_gather_latency,
        pre_weights_all_gather_latency / gradient_accumulation_steps,
    )
    post_weights_all_gather_latency = predict_gpu_gpu_comm_latency(
        op_name="all_gather",
        mbytes=mem_post_layer_full_weights,
        **_pre_post_dp_comm_kwargs,
    )
    post_weights_all_gather_latency = np.where(
        pre_post_dp_size == 1, 0, post_weights_all_gather_latency
    )
    post_weights_all_gather_latency_amortized = np.where(
        pre_post_wre_enabled == 1,
        post_weights_all_gather_latency,
        post_weights_all_gather_latency / gradient_accumulation_steps,
    )
    # Pre post layer grads all gathering
    pre_grad_sync_latency = predict_gpu_gpu_comm_latency(
        op_name="reduce_scatter",
        mbytes=mem_pre_layer_full_grads,
        **_pre_post_dp_comm_kwargs,
    )
    pre_grad_all_reduce_latency = predict_gpu_gpu_comm_latency(
        op_name="all_reduce",
        mbytes=mem_pre_layer_full_grads,
        **_pre_post_dp_comm_kwargs,
    )
    pre_grad_sync_latency = np.where(pre_post_dp_size == 1, 0, pre_grad_sync_latency)
    pre_grad_sync_latency_amortized = np.where(
        pre_post_gre_enabled == 1,
        pre_grad_sync_latency,
        pre_grad_sync_latency / gradient_accumulation_steps,
    )
    pre_grad_sync_latency_amortized = np.where(
        pre_post_ore_enabled == 0,
        pre_grad_all_reduce_latency / gradient_accumulation_steps,
        pre_grad_sync_latency_amortized,
    )

    post_grad_sync_latency = predict_gpu_gpu_comm_latency(
        op_name="reduce_scatter",
        mbytes=mem_post_layer_full_grads,
        **_pre_post_dp_comm_kwargs,
    )
    post_grad_all_reduce_latency = predict_gpu_gpu_comm_latency(
        op_name="all_reduce",
        mbytes=mem_post_layer_full_grads,
        **_pre_post_dp_comm_kwargs,
    )
    post_grad_sync_latency = np.where(pre_post_dp_size == 1, 0, post_grad_sync_latency)
    post_grad_sync_latency_amortized = np.where(
        pre_post_gre_enabled == 1,
        post_grad_sync_latency,
        post_grad_sync_latency / gradient_accumulation_steps,
    )
    post_grad_sync_latency_amortized = np.where(
        pre_post_ore_enabled == 0,
        post_grad_all_reduce_latency / gradient_accumulation_steps,
        post_grad_sync_latency_amortized,
    )

    # Pre-post layer end
    # ==================================================================================================

    # Get the information of the block layer
    batch_size = stage_strategies[:, 0]
    dp_size = stage_strategies[:, 1]
    tp_size = stage_strategies[:, 2]
    wre_enabled = stage_strategies[:, 3]
    gre_enabled = stage_strategies[:, 4]
    ore_enabled = stage_strategies[:, 5]
    wo_ratio = stage_strategies[:, 6]
    go_ratio = stage_strategies[:, 7]
    oo_ratio = stage_strategies[:, 8]
    ac_ratio = stage_strategies[:, 9]
    wre_size = np.where(wre_enabled == 1, dp_size, 1)
    gre_size = np.where(gre_enabled == 1, dp_size, 1)
    ore_size = np.where(ore_enabled == 1, dp_size, 1)

    has_state_swapping = np.any(
        (
            wo_ratio > 0,
            go_ratio > 0,
            oo_ratio > 0,
        ),
        axis=0,
    )
    has_activation_swapping = ac_ratio > 0
    has_wre = wre_enabled
    has_gre = gre_enabled

    # Consider the fragmentation
    frag_factor = 0.0
    if hasattr(config.model, "hidden_size") and hasattr(
        config.training, "max_sequence_length"
    ):
        hidden_size = config.model.hidden_size
        max_sequence_length = config.training.max_sequence_length
        frag_factor += max(0, hidden_size / 2048 - 1) * 0.005
        frag_factor += max(0, max_sequence_length / 2048 - 1) * 0.045

    # Memory factors
    memory_factor = np.ones_like(batch_size, dtype=float)
    memory_factor += frag_factor
    # Consider state swapping
    memory_factor += np.where(wo_ratio > 0, 0.025, 0)
    memory_factor += np.where(go_ratio > 0, 0.025, 0)
    memory_factor += np.where(oo_ratio > 0, 0.03, 0)
    # Consider activation swapping
    memory_factor += np.where(has_activation_swapping, 0.04, 0)
    # Consider has wre and gre
    memory_factor += np.where(has_wre, 0.025, 0)
    memory_factor += np.where(has_gre, 0.025, 0)
    # Consider TP
    memory_factor += np.where(tp_size > 1, 0.04, 0)
    # Consider the large batch size
    memory_factor += np.where(batch_size >= 4, 0.01, 0)
    # memory_factor += 0.025  # TMP
    # Consider the pp memory
    # memory_factor += 0.01  # TMP
    memory_factor_for_pp = memory_factor + 0.025


    # Calculate the global constant buffer memory (which means only onces for a stage-module)
    mem_global_constant_buffer = 0
    if not config.model.use_flash_attn:
        mem_attn_buffer = (
            2
            * batch_size
            * config.model.num_attention_heads
            / tp_size
            * (config.training.max_sequence_length**2)
            / (1024**2)
        )
        mem_global_constant_buffer += mem_attn_buffer

    # * memory
    mem_layer_full_weights = get_concrete_mem_full_states(
        block_layer_info, item="weights", tp_size=tp_size
    )
    mem_layer_full_grads = get_concrete_mem_full_states(
        block_layer_info, item="grads", tp_size=tp_size
    )
    mem_layer_full_opts = get_concrete_mem_full_states(
        block_layer_info, item="opts", tp_size=tp_size
    )
    mem_layer_partial_weights = get_concrete_mem_partial_states(
        mem_layer_full_weights,
        sharding_size=wre_size,
        offloading_ratio=wo_ratio,
    )
    mem_layer_partial_grads = get_concrete_mem_partial_states(
        mem_layer_full_grads,
        sharding_size=gre_size,
        offloading_ratio=go_ratio,
    )
    mem_layer_partial_opts = get_concrete_mem_partial_states(
        mem_layer_full_opts,
        sharding_size=ore_size,
        offloading_ratio=oo_ratio,
    )
    # Because the saved tensors for tp is redundant, we can use the
    # redundancy sharding technique to reduce the memory usage
    # TODO(zhanda): the activation offloading ratio may not be directly applied to this term
    # because we have pre-fetch and thus activations of one layer should be full
    _raw_mem_layer_saved_with_ckpt = get_concrete_mem_saved(
        block_layer_info,
        ckpt=True,
        batch_size_per_device=batch_size,
        tp_size=tp_size,
    )
    PARTITION_SAVED_TENSORS = False
    if PARTITION_SAVED_TENSORS:
        _raw_mem_layer_saved_with_ckpt_div_by_tp = (
            _raw_mem_layer_saved_with_ckpt / tp_size
        )
    else:
        _raw_mem_layer_saved_with_ckpt_div_by_tp = _raw_mem_layer_saved_with_ckpt
    mem_layer_saved_with_ckpt = _raw_mem_layer_saved_with_ckpt_div_by_tp * (
        1 - ac_ratio
    )
    mem_layer_saved_with_ckpt_delta_for_peak = (
        _raw_mem_layer_saved_with_ckpt_div_by_tp * ac_ratio
    )
    _raw_mem_layer_saved_without_ckpt = get_concrete_mem_saved(
        block_layer_info,
        ckpt=False,
        batch_size_per_device=batch_size,
        tp_size=tp_size,
    )
    mem_layer_saved_without_ckpt = _raw_mem_layer_saved_without_ckpt * (1 - ac_ratio)
    mem_layer_saved_without_ckpt_delta_for_peak = (
        _raw_mem_layer_saved_without_ckpt * ac_ratio
    )
    mem_layer_fwd_peak_with_ckpt = get_concrete_mem_peak(
        block_layer_info,
        phase_type="fwd",
        batch_size_per_device=batch_size,
        tp_size=tp_size,
        ckpt=True,
    )
    mem_layer_fwd_peak_without_ckpt = get_concrete_mem_peak(
        block_layer_info,
        phase_type="fwd",
        batch_size_per_device=batch_size,
        tp_size=tp_size,
        ckpt=False,
    )
    mem_layer_bwd_peak = get_concrete_mem_peak(
        block_layer_info,
        phase_type="bwd",
        batch_size_per_device=batch_size,
        tp_size=tp_size,
    )
    mem_layer_output = get_concrete_mem_output(
        block_layer_info,
        batch_size_per_device=batch_size,
        tp_size=tp_size,
    )
    # Latency of pipeline transfer
    _p2p_inter_gpu = num_gpus_per_node == config.hardware.num_gpus_per_node
    latency_p2p_output = (
        predict_gpu_gpu_comm_latency(
            op_name="p2p",
            mbytes=mem_layer_output,
            inter_size=1 if _p2p_inter_gpu else None,
            intra_size=None if _p2p_inter_gpu else 1,
        )
        * 2
    )
    # logger.debug(f"Mem layer opts: {mem_layer_full_opts}")
    # logger.debug(f"Mem pre-layer opts: {mem_pre_layer_full_opts}")
    # logger.debug(f"Mem post-layer opts: {mem_post_layer_full_opts}")

    # * latency
    dp_comm_kwargs = {
        "inter_size": np.where(dp_size > num_nodes, num_nodes, dp_size),
        "intra_size": np.where(dp_size > num_nodes, dp_size // num_nodes, 1),
    }
    tp_comm_kwargs = {
        "inter_size": np.where(
            tp_size > num_gpus_per_node, tp_size // num_gpus_per_node, 1
        ),
        "intra_size": np.where(tp_size > num_gpus_per_node, num_gpus_per_node, tp_size),
    }
    latencies_layer_exec = get_concrete_latency_exec(
        block_layer_info,
        batch_size_per_device=batch_size,
        tp_size=tp_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        **tp_comm_kwargs,
        **gpu_gpu_bw_kwargs,
    )
    latency_layer_exec_fwd = latencies_layer_exec[0]
    latency_layer_exec_bwd_without_ckpt = latencies_layer_exec[1]
    latency_layer_exec_extra_fwd = latencies_layer_exec[2]
    latency_layer_exec_extra_bwd = latencies_layer_exec[3]

    # Early exit if only doing op profiling
    if os.environ.get("MIST_OP_PROFILING", "0") == "1":
        print(f"Profiling Op, and early exit")
        return (
            (costs_stable, costs_delta, solutions),
            (costs_stable_with_pre, costs_delta_with_pre, solutions_with_pre),
            (costs_stable_with_post, costs_delta_with_post, solutions_with_post),
            (costs_stable_no_pp, costs_delta_no_pp, solutions_no_pp),
        )

    # For tp != 1, because the saved tensors are sharded, we need to
    # all gather the saved tensors
    if PARTITION_SAVED_TENSORS:
        latency_layer_bwd_all_gather_saved_with_ckpt = predict_gpu_gpu_comm_latency(
            op_name="all_gather",
            mbytes=_raw_mem_layer_saved_with_ckpt,
            **tp_comm_kwargs,
        )
    else:
        latency_layer_bwd_all_gather_saved_with_ckpt = np.array([0.0])
    latency_layer_exec_bwd_with_ckpt = (
        latency_layer_exec_fwd + latency_layer_exec_bwd_without_ckpt
    )
    # Non overlapped communication: Redundancy elimination + Offloading
    # 1. Offloading
    latency_layer_swap_out_weights = predict_gpu_cpu_comm_latency(
        mbytes=mem_layer_full_weights / wre_size * wo_ratio
    )
    latency_layer_swap_in_weights = predict_cpu_gpu_comm_latency(
        mbytes=mem_layer_full_weights / wre_size * wo_ratio
    )
    latency_layer_swap_out_grads = predict_gpu_cpu_comm_latency(
        mbytes=mem_layer_full_grads / gre_size * go_ratio
    )
    latency_layer_swap_in_grads = predict_cpu_gpu_comm_latency(
        mbytes=mem_layer_full_grads / gre_size * go_ratio
    )
    latency_layer_swap_out_opt_states = predict_gpu_cpu_comm_latency(
        mbytes=mem_layer_full_opts / ore_size * oo_ratio
    )
    latency_layer_swap_in_opt_states = predict_cpu_gpu_comm_latency(
        mbytes=mem_layer_full_opts / ore_size * oo_ratio
    )
    latency_layer_swap_out_activations_with_ckpt = predict_gpu_cpu_comm_latency(
        mbytes=_raw_mem_layer_saved_with_ckpt_div_by_tp * ac_ratio
    )
    latency_layer_swap_in_activations_with_ckpt = predict_cpu_gpu_comm_latency(
        mbytes=_raw_mem_layer_saved_with_ckpt_div_by_tp * ac_ratio
    )
    latency_layer_swap_out_activations_without_ckpt = predict_gpu_cpu_comm_latency(
        mbytes=_raw_mem_layer_saved_without_ckpt * ac_ratio
    )
    latency_layer_swap_in_activations_without_ckpt = predict_cpu_gpu_comm_latency(
        mbytes=_raw_mem_layer_saved_without_ckpt * ac_ratio
    )

    # 2. Communication
    latency_layer_all_gather_weights = predict_gpu_gpu_comm_latency(
        op_name="all_gather",
        mbytes=mem_layer_full_weights,
        **dp_comm_kwargs,
    )
    latency_layer_reduce_scatter_grads = predict_gpu_gpu_comm_latency(
        op_name="reduce_scatter",
        mbytes=mem_layer_full_grads,
        **dp_comm_kwargs,
    )
    latency_layer_all_reduce_grads = predict_gpu_gpu_comm_latency(
        op_name="all_reduce",
        mbytes=mem_layer_full_grads,
        **dp_comm_kwargs,
    )

    # 3. Utils
    zeros = np.zeros_like(latency_layer_exec_fwd)

    def get_layer_critical_latency(
        exec, gpu_gpu_comm, gpu_to_cpu_comm, cpu_to_gpu_comm
    ):
        return interference_estimate_for_one_group(
            C=exec,
            G2G=gpu_gpu_comm,
            C2G=cpu_to_gpu_comm,
            G2C=gpu_to_cpu_comm,
            params=interference_model_params,
        )

    # ==================================================================================================
    # Fwd Comm
    # the first iteration must
    #   1) load fp32 weights and fp32 opt states
    #   2) load fp16 grads
    #   3) all gather weights if ore enabled
    #   4) swap out fp32 weights and fp32 opt states
    #   5) swap out fp16 weights
    _latency_layer_gpu_to_cpu_fwd_first_micro_batch_exclude_act = (
        latency_layer_swap_out_opt_states + latency_layer_swap_out_weights
    )
    _latency_layer_cpu_to_gpu_fwd_first_micro_batch_exclude_act = (
        latency_layer_swap_in_opt_states + latency_layer_swap_in_grads
    )
    _latency_layer_gpu_gpu_fwd_first_micro_batch_exclude_act = (
        latency_layer_all_gather_weights * ore_enabled
    )
    # not the first iteration
    #   1) load weights
    #   2) all gather weights if wre enabled
    _latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act = np.zeros_like(
        latency_layer_swap_out_weights
    )
    _latency_layer_cpu_to_gpu_fwd_not_first_micro_batch_exclude_act = (
        latency_layer_swap_in_weights
    )
    _latency_layer_gpu_gpu_fwd_not_first_micro_batch_exclude_act = (
        latency_layer_all_gather_weights * wre_enabled
    )
    # ==================================================================================================
    # First MicroBatch - Without CKPT
    latency_layer_critical_fwd_first_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_fwd - latency_layer_exec_extra_fwd,
            gpu_gpu_comm=_latency_layer_gpu_gpu_fwd_first_micro_batch_exclude_act,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_first_micro_batch_exclude_act
            + latency_layer_swap_out_activations_without_ckpt,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_fwd_first_micro_batch_exclude_act,
        )
        + latency_layer_exec_extra_fwd
    )
    latency_first_layer_critical_fwd_first_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_fwd - latency_layer_exec_extra_fwd,
            gpu_gpu_comm=_latency_layer_gpu_gpu_fwd_first_micro_batch_exclude_act,
            gpu_to_cpu_comm=zeros,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_fwd_first_micro_batch_exclude_act,
        )
        + latency_layer_exec_extra_fwd
    )
    latency_last_layer_critical_fwd_first_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_fwd - latency_layer_exec_extra_fwd,
            gpu_gpu_comm=zeros,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_first_micro_batch_exclude_act
            + latency_layer_swap_out_activations_without_ckpt,
            cpu_to_gpu_comm=zeros,
        )
        + latency_layer_exec_extra_fwd
    )
    latency_extra_pre_fwd_first_micro_batch_without_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=_latency_layer_gpu_gpu_fwd_first_micro_batch_exclude_act,
        gpu_to_cpu_comm=zeros,
        cpu_to_gpu_comm=(
            _latency_layer_cpu_to_gpu_fwd_first_micro_batch_exclude_act
            if enable_advanced_opt_in_first_block
            else zeros
        ),
    )
    latency_extra_post_fwd_first_micro_batch_without_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=zeros,
        gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_first_micro_batch_exclude_act
        + latency_layer_swap_out_activations_without_ckpt,
        cpu_to_gpu_comm=zeros,
    )
    # First MicroBatch - With CKPT
    latency_layer_critical_fwd_first_micro_batch_with_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_fwd - latency_layer_exec_extra_fwd,
            gpu_gpu_comm=_latency_layer_gpu_gpu_fwd_first_micro_batch_exclude_act,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_first_micro_batch_exclude_act
            + latency_layer_swap_out_activations_with_ckpt,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_fwd_first_micro_batch_exclude_act,
        )
        + latency_layer_exec_extra_fwd
    )
    latency_first_layer_critical_fwd_first_micro_batch_with_ckpt = (
        latency_first_layer_critical_fwd_first_micro_batch_without_ckpt
    )
    latency_last_layer_critical_fwd_first_micro_batch_with_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_fwd - latency_layer_exec_extra_fwd,
            gpu_gpu_comm=zeros,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_first_micro_batch_exclude_act
            + latency_layer_swap_out_activations_with_ckpt,
            cpu_to_gpu_comm=zeros,
        )
        + latency_layer_exec_extra_fwd
    )
    latency_extra_pre_fwd_first_micro_batch_with_ckpt = (
        latency_extra_pre_fwd_first_micro_batch_without_ckpt
    )
    latency_extra_post_fwd_first_micro_batch_with_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=zeros,
        gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_first_micro_batch_exclude_act
        + latency_layer_swap_out_activations_with_ckpt,
        cpu_to_gpu_comm=zeros,
    )
    # Not the first MicroBatch - Without CKPT
    latency_layer_critical_fwd_not_first_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_fwd - latency_layer_exec_extra_fwd,
            gpu_gpu_comm=_latency_layer_gpu_gpu_fwd_not_first_micro_batch_exclude_act,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act
            + latency_layer_swap_out_activations_without_ckpt,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_fwd_not_first_micro_batch_exclude_act,
        )
        + latency_layer_exec_extra_fwd
    )
    latency_first_layer_critical_fwd_not_first_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_fwd - latency_layer_exec_extra_fwd,
            gpu_gpu_comm=_latency_layer_gpu_gpu_fwd_not_first_micro_batch_exclude_act,
            gpu_to_cpu_comm=zeros,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_fwd_not_first_micro_batch_exclude_act,
        )
        + latency_layer_exec_extra_fwd
    )
    latency_last_layer_critical_fwd_not_first_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_fwd - latency_layer_exec_extra_fwd,
            gpu_gpu_comm=zeros,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act
            + latency_layer_swap_out_activations_without_ckpt,
            cpu_to_gpu_comm=zeros,
        )
        + latency_layer_exec_extra_fwd
    )
    latency_extra_pre_fwd_not_first_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=zeros,
            gpu_gpu_comm=(
                _latency_layer_gpu_gpu_fwd_not_first_micro_batch_exclude_act
                if enable_advanced_opt_in_first_block
                else zeros
            ),
            gpu_to_cpu_comm=zeros,
            cpu_to_gpu_comm=(
                _latency_layer_cpu_to_gpu_fwd_not_first_micro_batch_exclude_act
                if enable_advanced_opt_in_first_block
                else zeros
            ),
        )
    )
    latency_extra_post_fwd_not_first_micro_batch_without_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=zeros,
        gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act
        + latency_layer_swap_out_activations_without_ckpt,
        cpu_to_gpu_comm=zeros,
    )
    # Not the first MicroBatch - With CKPT
    latency_layer_critical_fwd_not_first_micro_batch_with_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_fwd - latency_layer_exec_extra_fwd,
            gpu_gpu_comm=_latency_layer_gpu_gpu_fwd_not_first_micro_batch_exclude_act,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act
            + latency_layer_swap_out_activations_with_ckpt,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_fwd_not_first_micro_batch_exclude_act,
        )
        + latency_layer_exec_extra_fwd
    )
    latency_first_layer_critical_fwd_not_first_micro_batch_with_ckpt = (
        latency_first_layer_critical_fwd_not_first_micro_batch_without_ckpt
    )
    latency_last_layer_critical_fwd_not_first_micro_batch_with_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_fwd - latency_layer_exec_extra_fwd,
            gpu_gpu_comm=zeros,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act
            + latency_layer_swap_out_activations_with_ckpt,
            cpu_to_gpu_comm=zeros,
        )
        + latency_layer_exec_extra_fwd
    )
    latency_extra_pre_fwd_not_first_micro_batch_with_ckpt = (
        latency_extra_pre_fwd_not_first_micro_batch_without_ckpt
    )
    latency_extra_post_fwd_not_first_micro_batch_with_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=zeros,
        gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act
        + latency_layer_swap_out_activations_with_ckpt,
        cpu_to_gpu_comm=zeros,
    )
    # ==================================================================================================
    # Extra with Post
    # latency_last_layer_critical_fwd_not_first_micro_batch_without_ckpt_with_post_layer = get_layer_critical_latency(
    #     exec=latency_layer_exec_fwd,
    #     gpu_gpu_comm=post_grad_sync_latency,
    #     gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act
    #     + latency_layer_swap_out_activations_with_ckpt,
    #     cpu_to_gpu_comm=zeros,
    # )
    # latency_last_layer_critical_fwd_not_first_micro_batch_with_ckpt = get_layer_critical_latency(
    #     exec=latency_layer_exec_fwd,
    #     gpu_gpu_comm=zeros,
    #     gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act
    #     + latency_layer_swap_out_activations_with_ckpt,
    #     cpu_to_gpu_comm=zeros,
    # )
    # ==================================================================================================
    # Bwd Comm
    # the last iteration must
    #   1) load weights
    #   2) load grads
    #   3) all gather weights if wre enabled
    #   4) reduce scatter grads
    #   5) swap out grads
    _latency_layer_gpu_to_cpu_bwd_last_micro_batch_exclude_act = (
        latency_layer_swap_out_grads
    )
    _latency_layer_cpu_to_gpu_bwd_last_micro_batch_exclude_act = (
        latency_layer_swap_in_weights + latency_layer_swap_in_grads
    )
    _latency_layer_gpu_gpu_bwd_last_micro_batch_exclude_act = (
        latency_layer_all_gather_weights * wre_enabled
        + latency_layer_reduce_scatter_grads * ore_enabled
        + latency_layer_all_reduce_grads * (1 - ore_enabled)
    )
    # not the last iteration
    #   1) load weights
    #   2) load grads
    #   3) all gather weights if wre enabled
    #   4) reduce scatter grads if gre enabled
    #   5) swap out grads
    _latency_layer_gpu_to_cpu_bwd_not_last_micro_batch_exclude_act = (
        latency_layer_swap_out_grads
    )
    _latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act = (
        latency_layer_swap_in_weights + latency_layer_swap_in_grads
    )
    _latency_layer_gpu_gpu_bwd_not_last_micro_batch_exclude_act = (
        latency_layer_all_gather_weights * wre_enabled
        + latency_layer_reduce_scatter_grads * gre_enabled
    )
    # ==================================================================================================
    # Bwd Critical
    # Last MicroBatch - Without CKPT
    latency_layer_exec_extra_fwd_and_bwd = (
        latency_layer_exec_extra_fwd + latency_layer_exec_extra_bwd
    )
    latency_layer_critical_bwd_last_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_without_ckpt - latency_layer_exec_extra_bwd,
            gpu_gpu_comm=_latency_layer_gpu_gpu_bwd_last_micro_batch_exclude_act,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_bwd_last_micro_batch_exclude_act,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_last_micro_batch_exclude_act
            + latency_layer_swap_in_activations_without_ckpt,
        )
        + latency_layer_exec_extra_bwd
    )
    latency_last_layer_critical_bwd_last_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_without_ckpt - latency_layer_exec_extra_bwd,
            gpu_gpu_comm=latency_layer_all_gather_weights * wre_enabled,
            gpu_to_cpu_comm=zeros,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_last_micro_batch_exclude_act
            + latency_layer_swap_in_activations_without_ckpt,
        )
        + latency_layer_exec_extra_bwd
    )
    latency_first_layer_critical_bwd_last_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_without_ckpt - latency_layer_exec_extra_bwd,
            gpu_gpu_comm=latency_layer_reduce_scatter_grads * ore_enabled
            + latency_layer_all_reduce_grads * (1 - ore_enabled),
            gpu_to_cpu_comm=(
                _latency_layer_gpu_to_cpu_bwd_last_micro_batch_exclude_act
                if enable_advanced_opt_in_first_block
                else zeros
            ),
            cpu_to_gpu_comm=zeros,
        )
        + latency_layer_exec_extra_bwd
    )
    latency_extra_pre_bwd_last_micro_batch_without_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=latency_layer_all_gather_weights * wre_enabled,
        gpu_to_cpu_comm=zeros,
        cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_last_micro_batch_exclude_act
        + latency_layer_swap_in_activations_without_ckpt,
    )
    latency_extra_post_bwd_last_micro_batch_without_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=latency_layer_reduce_scatter_grads * ore_enabled
        + latency_layer_all_reduce_grads * (1 - ore_enabled),
        gpu_to_cpu_comm=(
            _latency_layer_gpu_to_cpu_bwd_last_micro_batch_exclude_act
            if enable_advanced_opt_in_first_block
            else zeros
        ),
        cpu_to_gpu_comm=zeros,
    )
    # Last MicroBatch - With CKPT
    latency_layer_critical_bwd_last_micro_batch_with_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_with_ckpt
            - latency_layer_exec_extra_fwd_and_bwd,
            gpu_gpu_comm=_latency_layer_gpu_gpu_bwd_last_micro_batch_exclude_act
            + latency_layer_bwd_all_gather_saved_with_ckpt,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_bwd_last_micro_batch_exclude_act,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_last_micro_batch_exclude_act
            + latency_layer_swap_in_activations_with_ckpt,
        )
        + latency_layer_exec_extra_fwd_and_bwd
    )
    latency_last_layer_critical_bwd_last_micro_batch_with_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_with_ckpt
            - latency_layer_exec_extra_fwd_and_bwd,
            gpu_gpu_comm=latency_layer_all_gather_weights * wre_enabled,
            gpu_to_cpu_comm=zeros,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_last_micro_batch_exclude_act
            + latency_layer_swap_in_activations_with_ckpt,
        )
        + latency_layer_exec_extra_fwd_and_bwd
    )
    latency_first_layer_critical_bwd_last_micro_batch_with_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_with_ckpt
            - latency_layer_exec_extra_fwd_and_bwd,
            gpu_gpu_comm=np.where(
                ore_enabled == 1,
                latency_layer_reduce_scatter_grads,
                latency_layer_all_reduce_grads,
            ),
            gpu_to_cpu_comm=(
                _latency_layer_gpu_to_cpu_bwd_last_micro_batch_exclude_act
                if enable_advanced_opt_in_first_block
                else zeros
            ),
            cpu_to_gpu_comm=zeros,
        )
        + latency_layer_exec_extra_fwd_and_bwd
    )
    latency_extra_pre_bwd_last_micro_batch_with_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=latency_layer_all_gather_weights * wre_enabled,
        gpu_to_cpu_comm=zeros,
        cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_last_micro_batch_exclude_act
        + latency_layer_swap_in_activations_with_ckpt,
    )
    latency_extra_post_bwd_last_micro_batch_with_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=np.where(
            ore_enabled == 1,
            latency_layer_reduce_scatter_grads,
            latency_layer_all_reduce_grads,
        ),
        gpu_to_cpu_comm=(
            _latency_layer_gpu_to_cpu_bwd_last_micro_batch_exclude_act
            if enable_advanced_opt_in_first_block
            else zeros
        ),
        cpu_to_gpu_comm=zeros,
    )

    # Not Last MicroBatch - Without CKPT
    latency_layer_critical_bwd_not_last_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_without_ckpt - latency_layer_exec_extra_bwd,
            gpu_gpu_comm=_latency_layer_gpu_gpu_bwd_not_last_micro_batch_exclude_act,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_bwd_not_last_micro_batch_exclude_act,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act
            + latency_layer_swap_in_activations_without_ckpt,
        )
        + latency_layer_exec_extra_bwd
    )
    latency_last_layer_critical_bwd_not_last_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_without_ckpt - latency_layer_exec_extra_bwd,
            gpu_gpu_comm=latency_layer_all_gather_weights * wre_enabled,
            gpu_to_cpu_comm=zeros,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act
            + latency_layer_swap_in_activations_without_ckpt,
        )
        + latency_layer_exec_extra_bwd
    )
    latency_first_layer_critical_bwd_not_last_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_without_ckpt - latency_layer_exec_extra_bwd,
            gpu_gpu_comm=latency_layer_reduce_scatter_grads * gre_enabled,
            gpu_to_cpu_comm=(
                _latency_layer_gpu_to_cpu_bwd_not_last_micro_batch_exclude_act
                if enable_advanced_opt_in_first_block
                else zeros
            ),
            cpu_to_gpu_comm=zeros,
        )
        + latency_layer_exec_extra_bwd
    )
    latency_extra_pre_bwd_not_last_micro_batch_without_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=latency_layer_all_gather_weights * wre_enabled,
        gpu_to_cpu_comm=zeros,
        cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act
        + latency_layer_swap_in_activations_without_ckpt,
    )
    latency_extra_post_bwd_not_last_micro_batch_without_ckpt = (
        get_layer_critical_latency(
            exec=zeros,
            gpu_gpu_comm=latency_layer_reduce_scatter_grads * gre_enabled,
            gpu_to_cpu_comm=(
                _latency_layer_gpu_to_cpu_bwd_not_last_micro_batch_exclude_act
                if enable_advanced_opt_in_first_block
                else zeros
            ),
            cpu_to_gpu_comm=zeros,
        )
    )
    # Last MicroBatch - With CKPT
    latency_layer_critical_bwd_not_last_micro_batch_with_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_with_ckpt
            - latency_layer_exec_extra_fwd_and_bwd,
            gpu_gpu_comm=_latency_layer_gpu_gpu_bwd_not_last_micro_batch_exclude_act
            + latency_layer_bwd_all_gather_saved_with_ckpt,
            gpu_to_cpu_comm=_latency_layer_gpu_to_cpu_bwd_not_last_micro_batch_exclude_act,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act
            + latency_layer_swap_in_activations_with_ckpt,
        )
        + latency_layer_exec_extra_fwd_and_bwd
    )
    latency_last_layer_critical_bwd_not_last_micro_batch_with_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_with_ckpt
            - latency_layer_exec_extra_fwd_and_bwd,
            gpu_gpu_comm=latency_layer_all_gather_weights * wre_enabled,
            gpu_to_cpu_comm=zeros,
            cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act
            + latency_layer_swap_in_activations_with_ckpt,
        )
        + latency_layer_exec_extra_fwd_and_bwd
    )
    latency_first_layer_critical_bwd_not_last_micro_batch_with_ckpt = (
        get_layer_critical_latency(
            exec=latency_layer_exec_bwd_with_ckpt
            - latency_layer_exec_extra_fwd_and_bwd,
            gpu_gpu_comm=latency_layer_reduce_scatter_grads * gre_enabled,
            gpu_to_cpu_comm=(
                _latency_layer_gpu_to_cpu_bwd_not_last_micro_batch_exclude_act
                if enable_advanced_opt_in_first_block
                else zeros
            ),
            cpu_to_gpu_comm=zeros,
        )
        + latency_layer_exec_extra_fwd_and_bwd
    )
    latency_extra_pre_bwd_not_last_micro_batch_with_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=latency_layer_all_gather_weights * wre_enabled,
        gpu_to_cpu_comm=zeros,
        cpu_to_gpu_comm=_latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act
        + latency_layer_swap_in_activations_with_ckpt,
    )
    latency_extra_post_bwd_not_last_micro_batch_with_ckpt = get_layer_critical_latency(
        exec=zeros,
        gpu_gpu_comm=latency_layer_reduce_scatter_grads * gre_enabled,
        gpu_to_cpu_comm=(
            _latency_layer_gpu_to_cpu_bwd_not_last_micro_batch_exclude_act
            if enable_advanced_opt_in_first_block
            else zeros
        ),
        cpu_to_gpu_comm=zeros,
    )
    # ==================================================================================================
    # Combined Critical
    # Without ckpt
    # FIXME(zhanda): check whether this works for ZeRO-2 and ZeRO-3
    assert isinstance(
        gradient_accumulation_steps, Number
    ), f"gradient_accumulation_steps - type: {type(gradient_accumulation_steps)}"
    if gradient_accumulation_steps == 1:
        # Without ckpt
        latency_layer_critical_stable_without_ckpt = (
            latency_layer_critical_fwd_first_micro_batch_without_ckpt
            + latency_layer_critical_bwd_last_micro_batch_without_ckpt
        )
        latency_pre_fwd_and_post_bwd_and_first_layer_stable_without_ckpt = (
            latency_extra_pre_fwd_first_micro_batch_without_ckpt
            + latency_extra_post_bwd_last_micro_batch_without_ckpt
            + latency_first_layer_critical_fwd_first_micro_batch_without_ckpt
            + latency_first_layer_critical_bwd_last_micro_batch_without_ckpt
        )
        latency_post_fwd_and_pre_bwd_and_last_layer_stable_without_ckpt = (
            latency_extra_post_fwd_first_micro_batch_without_ckpt
            + latency_extra_pre_bwd_last_micro_batch_without_ckpt
            + latency_last_layer_critical_fwd_first_micro_batch_without_ckpt
            + latency_last_layer_critical_bwd_last_micro_batch_without_ckpt
        )
        # With ckpt
        latency_layer_critical_stable_with_ckpt = (
            latency_layer_critical_fwd_first_micro_batch_with_ckpt
            + latency_layer_critical_bwd_last_micro_batch_with_ckpt
        )
        latency_pre_fwd_and_post_bwd_and_first_layer_stable_with_ckpt = (
            latency_extra_pre_fwd_first_micro_batch_with_ckpt
            + latency_extra_post_bwd_last_micro_batch_with_ckpt
            + latency_first_layer_critical_fwd_first_micro_batch_with_ckpt
            + latency_first_layer_critical_bwd_last_micro_batch_with_ckpt
        )
        latency_post_fwd_and_pre_bwd_and_last_layer_stable_with_ckpt = (
            latency_extra_post_fwd_first_micro_batch_with_ckpt
            + latency_extra_pre_bwd_last_micro_batch_with_ckpt
            + latency_last_layer_critical_fwd_first_micro_batch_with_ckpt
            + latency_last_layer_critical_bwd_last_micro_batch_with_ckpt
        )
        # Delta
        latency_layer_critical_delta_without_ckpt = 0
        latency_pre_fwd_and_post_bwd_and_first_layer_delta_without_ckpt = 0
        latency_post_fwd_and_pre_bwd_and_last_layer_delta_without_ckpt = 0
        # With ckpt
        latency_layer_critical_delta_with_ckpt = 0
        latency_pre_fwd_and_post_bwd_and_first_layer_delta_with_ckpt = 0
        latency_post_fwd_and_pre_bwd_and_last_layer_delta_with_ckpt = 0

    elif gradient_accumulation_steps > 1:
        # Without CKPT - Stable
        latency_layer_critical_stable_without_ckpt = (
            latency_layer_critical_fwd_not_first_micro_batch_without_ckpt
            + latency_layer_critical_bwd_not_last_micro_batch_without_ckpt
        )
        latency_pre_fwd_and_post_bwd_and_first_layer_stable_without_ckpt = (
            latency_extra_pre_fwd_not_first_micro_batch_without_ckpt
            + latency_extra_post_bwd_not_last_micro_batch_without_ckpt
            + latency_first_layer_critical_fwd_not_first_micro_batch_without_ckpt
            + latency_first_layer_critical_bwd_not_last_micro_batch_without_ckpt
        )
        latency_post_fwd_and_pre_bwd_and_last_layer_stable_without_ckpt = (
            latency_extra_post_fwd_not_first_micro_batch_without_ckpt
            + latency_extra_pre_bwd_not_last_micro_batch_without_ckpt
            + latency_last_layer_critical_fwd_not_first_micro_batch_without_ckpt
            + latency_last_layer_critical_bwd_not_last_micro_batch_without_ckpt
        )
        # Without CKPT - Delta
        latency_layer_critical_delta_without_ckpt = (
            latency_layer_critical_fwd_first_micro_batch_without_ckpt
            + latency_layer_critical_bwd_last_micro_batch_without_ckpt
            - latency_layer_critical_stable_without_ckpt
        )
        latency_pre_fwd_and_post_bwd_and_first_layer_delta_without_ckpt = (
            latency_extra_pre_fwd_first_micro_batch_without_ckpt
            + latency_extra_post_bwd_last_micro_batch_without_ckpt
            + latency_first_layer_critical_fwd_first_micro_batch_without_ckpt
            + latency_first_layer_critical_bwd_last_micro_batch_without_ckpt
            - latency_pre_fwd_and_post_bwd_and_first_layer_stable_without_ckpt
        )
        latency_post_fwd_and_pre_bwd_and_last_layer_delta_without_ckpt = (
            latency_extra_post_fwd_first_micro_batch_without_ckpt
            + latency_extra_pre_bwd_last_micro_batch_without_ckpt
            + latency_last_layer_critical_fwd_first_micro_batch_without_ckpt
            + latency_last_layer_critical_bwd_last_micro_batch_without_ckpt
            - latency_post_fwd_and_pre_bwd_and_last_layer_stable_without_ckpt
        )
        # With CKPT - Stable
        latency_layer_critical_stable_with_ckpt = (
            latency_layer_critical_fwd_not_first_micro_batch_with_ckpt
            + latency_layer_critical_bwd_not_last_micro_batch_with_ckpt
        )
        latency_pre_fwd_and_post_bwd_and_first_layer_stable_with_ckpt = (
            latency_extra_pre_fwd_not_first_micro_batch_with_ckpt
            + latency_extra_post_bwd_not_last_micro_batch_with_ckpt
            + latency_first_layer_critical_fwd_not_first_micro_batch_with_ckpt
            + latency_first_layer_critical_bwd_not_last_micro_batch_with_ckpt
        )
        latency_post_fwd_and_pre_bwd_and_last_layer_stable_with_ckpt = (
            latency_extra_post_fwd_not_first_micro_batch_with_ckpt
            + latency_extra_pre_bwd_not_last_micro_batch_with_ckpt
            + latency_last_layer_critical_fwd_not_first_micro_batch_with_ckpt
            + latency_last_layer_critical_bwd_not_last_micro_batch_with_ckpt
        )
        # With CKPT - Delta
        latency_layer_critical_delta_with_ckpt = (
            latency_layer_critical_fwd_first_micro_batch_with_ckpt
            + latency_layer_critical_bwd_last_micro_batch_with_ckpt
            - latency_layer_critical_stable_with_ckpt
        )
        latency_pre_fwd_and_post_bwd_and_first_layer_delta_with_ckpt = (
            latency_extra_pre_fwd_first_micro_batch_with_ckpt
            + latency_extra_post_bwd_last_micro_batch_with_ckpt
            + latency_first_layer_critical_fwd_first_micro_batch_with_ckpt
            + latency_first_layer_critical_bwd_last_micro_batch_with_ckpt
            - latency_pre_fwd_and_post_bwd_and_first_layer_stable_with_ckpt
        )
        latency_post_fwd_and_pre_bwd_and_last_layer_delta_with_ckpt = (
            latency_extra_post_fwd_first_micro_batch_with_ckpt
            + latency_extra_pre_bwd_last_micro_batch_with_ckpt
            + latency_last_layer_critical_fwd_first_micro_batch_with_ckpt
            + latency_last_layer_critical_bwd_last_micro_batch_with_ckpt
            - latency_post_fwd_and_pre_bwd_and_last_layer_stable_with_ckpt
        )

    # latency_layer_critical_fwd_without_ckpt = (
    #     (latency_layer_critical_fwd_first_micro_batch_without_ckpt + latency_layer_extra_pre_fwd_first_micro_batch)
    #     + (gradient_accumulation_steps - 1)
    #     * (latency_layer_critical_fwd_not_first_micro_batch_without_ckpt + latency_layer_extra_pre_fwd_not_first_micro_batch)
    # ) / gradient_accumulation_steps
    # latency_layer_critical_bwd_without_ckpt = (
    #     (latency_layer_critical_bwd_last_micro_batch_without_ckpt + latency_layer_extra_post_bwd_last_micro_batch)
    #     + (gradient_accumulation_steps - 1)
    #     * (latency_layer_critical_bwd_not_last_micro_batch_without_ckpt + latency_layer_extra_post_bwd_not_last_micro_batch)
    # ) / gradient_accumulation_steps
    # latency_layer_critical_without_ckpt = (
    #     latency_layer_critical_fwd_without_ckpt
    #     + latency_layer_critical_bwd_without_ckpt
    # )
    # With ckpt
    # latency_layer_critical_fwd_with_ckpt = (
    #     (latency_layer_critical_fwd_first_micro_batch_with_ckpt + latency_layer_extra_pre_fwd_first_micro_batch)
    #     + (gradient_accumulation_steps - 1)
    #     * (latency_layer_critical_fwd_not_first_micro_batch_with_ckpt + latency_layer_extra_pre_fwd_not_first_micro_batch)
    # ) / gradient_accumulation_steps
    # latency_layer_critical_bwd_with_ckpt = (
    #     (latency_layer_critical_bwd_last_micro_batch_with_ckpt + latency_layer_extra_post_bwd_last_micro_batch)
    #     + (gradient_accumulation_steps - 1)
    #     * (latency_layer_critical_bwd_not_last_micro_batch_with_ckpt + latency_layer_extra_post_bwd_not_last_micro_batch)
    # ) / gradient_accumulation_steps
    # latency_layer_critical_with_ckpt = (
    #     latency_layer_critical_fwd_with_ckpt + latency_layer_critical_bwd_with_ckpt
    # )
    # ==================================================================================================

    # Construct the vectorized num_layers and
    # add one more dimension to support the vectorized calculation
    num_layers = np.array(num_layers_candidates)
    num_layers = num_layers[:, np.newaxis]
    n = num_layers.shape[0]

    # Calculate the memories unrelated to the for-loop variables
    num_layers_minus_2 = np.maximum((num_layers - 2), 0)
    num_layers_minus_1 = np.maximum((num_layers - 1), 0)
    # mem_module_fwd_states = (
    #     2 * mem_layer_full_weights
    #     + num_layers_minus_2 * mem_layer_partial_weights
    #     + num_layers * (mem_layer_partial_grads + mem_layer_partial_opts)
    # )
    # ==========================================================================================
    # Before considering the memory buffer
    # mem_module_fwd_states = 2 * (
    #     mem_layer_full_weights
    #     + mem_layer_full_grads // wre_size
    #     + mem_layer_full_opts // ore_size
    # ) + num_layers_minus_2 * (
    #     mem_layer_partial_weights + mem_layer_partial_grads + mem_layer_partial_opts
    # )
    # mem_module_bwd_states = (
    #     2 * (mem_layer_full_weights + mem_layer_full_grads)
    #     + num_layers_minus_2 * (mem_layer_partial_weights + mem_layer_partial_grads)
    #     + num_layers * mem_layer_partial_opts
    # )
    # ==========================================================================================
    # After considering the memory buffer
    # Broadcast pre post info
    _pre_post_wre_enabled = np.broadcast_to(pre_post_wre_enabled, wre_size.shape)
    _pre_post_gre_enabled = np.broadcast_to(pre_post_gre_enabled, gre_size.shape)
    _mem_pre_layer_full_weights = np.broadcast_to(
        mem_pre_layer_full_weights, wre_size.shape
    )
    _mem_pre_layer_full_grads = np.broadcast_to(
        mem_pre_layer_full_grads, gre_size.shape
    )
    _mem_pre_layer_full_opts = np.broadcast_to(mem_pre_layer_full_opts, ore_size.shape)
    _mem_post_layer_full_weights = np.broadcast_to(
        mem_post_layer_full_weights, wre_size.shape
    )
    _mem_post_layer_full_grads = np.broadcast_to(
        mem_post_layer_full_grads, gre_size.shape
    )
    has_weight_swapping_or_sharding = np.logical_or.reduce(
        (
            wre_size > 1,
            _pre_post_wre_enabled == 1,
            wo_ratio > 0,
        ),
    )
    has_grad_swapping_or_sharding = np.logical_or.reduce(
        (
            gre_size > 1,
            _pre_post_gre_enabled == 1,
            go_ratio > 0,
        ),
    )
    mem_per_full_weight_buffer = np.where(
        has_weight_swapping_or_sharding,
        np.maximum.reduce(
            (
                _mem_pre_layer_full_weights,
                _mem_post_layer_full_weights,
                mem_layer_full_weights,
            )
        ),
        0,
    )
    mem_per_full_grad_buffer = np.where(
        has_grad_swapping_or_sharding,
        np.maximum.reduce(
            (
                _mem_pre_layer_full_grads,
                _mem_post_layer_full_grads,
                mem_layer_full_grads,
            )
        ),
        0,
    )
    # print(mem_per_full_weight_buffer.shape)
    # print(mem_per_full_grad_buffer.shape)
    # print(mem_layer_full_opts.shape)
    # print((mem_layer_full_opts // ore_size).shape)
    # print(mem_layer_partial_weights.shape)
    # print(mem_layer_partial_grads.shape)
    # print(mem_layer_partial_opts.shape)
    # exit()
    mem_module_fwd_states = (
        2 * (mem_per_full_weight_buffer + mem_per_full_grad_buffer)
        + mem_layer_full_opts // ore_size
        # Memory for the full weights to be all-gathers (on fp32)
        # + mem_layer_full_opts // 3 * (1 - 1 / ore_size)
        + num_layers * (mem_layer_partial_weights + mem_layer_partial_grads)
        + num_layers_minus_1 * mem_layer_partial_opts
    )
    mem_module_bwd_states = 2 * (
        mem_per_full_weight_buffer + mem_per_full_grad_buffer
    ) + num_layers * (
        mem_layer_partial_weights + mem_layer_partial_grads + mem_layer_partial_opts
    )
    # ==========================================================================================

    def get_latencies(num_layers, num_ckpt_layers):
        no_ckpt = num_ckpt_layers == 0
        all_ckpt = num_ckpt_layers == num_layers

        num_stable_ckpt_layers = np.clip(num_ckpt_layers - 1, 0, num_layers_minus_2)
        num_stable_non_ckpt_layers = num_layers_minus_2 - num_stable_ckpt_layers

        latencies_stable = (
            num_stable_ckpt_layers * latency_layer_critical_stable_with_ckpt
            + num_stable_non_ckpt_layers * latency_layer_critical_stable_without_ckpt
            +
            # Pre-fwd, post-bwd, and the first layer
            np.where(
                all_ckpt,
                latency_pre_fwd_and_post_bwd_and_first_layer_stable_with_ckpt,
                latency_pre_fwd_and_post_bwd_and_first_layer_stable_without_ckpt,
            )
            +
            # Post-fwd, pre-bwd, and the last layer
            np.where(
                no_ckpt,
                latency_post_fwd_and_pre_bwd_and_last_layer_stable_without_ckpt,
                latency_post_fwd_and_pre_bwd_and_last_layer_stable_with_ckpt,
            )
        )

        # Decouple the latencies
        latencies_delta = (
            num_stable_ckpt_layers * latency_layer_critical_delta_with_ckpt
            + num_stable_non_ckpt_layers * latency_layer_critical_delta_without_ckpt
            +
            # Pre-fwd, post-bwd, and the first layer
            np.where(
                all_ckpt,
                latency_pre_fwd_and_post_bwd_and_first_layer_delta_with_ckpt,
                latency_pre_fwd_and_post_bwd_and_first_layer_delta_without_ckpt,
            )
            +
            # Post-fwd, pre-bwd, and the last layer
            np.where(
                no_ckpt,
                latency_post_fwd_and_pre_bwd_and_last_layer_delta_without_ckpt,
                latency_post_fwd_and_pre_bwd_and_last_layer_delta_with_ckpt,
            )
        )
        # Expand the delta to the original shape
        latencies_delta = np.broadcast_to(latencies_delta, latencies_stable.shape)
        latencies_stable = latencies_stable.astype(np.float32)
        latencies_delta = latencies_delta.astype(np.float32)
        # Combine the stable and delta
        latencies = latencies_stable + latencies_delta / gradient_accumulation_steps
        return latencies, latencies_stable, latencies_delta

    for num_ckpt_layers in tqdm(
        num_ckpt_layers_candidates, disable=True, position=2, leave=False
    ):
        # flags
        valid_layer_mask = num_ckpt_layers <= num_layers.flatten()
        ckpt_layers_equals_num_layers = num_ckpt_layers == num_layers
        ckpt_layers_equals_zero = num_ckpt_layers == 0
        last_layer_ckpt = num_ckpt_layers > 0

        # calculate the latencies
        latency, latency_stable, latency_delta = get_latencies(
            num_layers, num_ckpt_layers
        )

        # stable and delta
        latency_stable_with_pre = (
            latency_stable
            + pre_fwd_latency
            + pre_bwd_latency
            + np.where(pre_post_wre_enabled == 1, pre_weights_all_gather_latency, 0)
            + np.where(pre_post_gre_enabled == 1, pre_grad_sync_latency, 0)
        )
        latency_delta_with_pre = (
            latency_delta
            + np.where(pre_post_wre_enabled == 1, 0, pre_weights_all_gather_latency)
            + np.where(pre_post_gre_enabled == 1, 0, pre_grad_sync_latency)
        )
        latency_with_pre = (
            latency_stable_with_pre
            + latency_delta_with_pre / gradient_accumulation_steps
        )
        # TODO(zhanda): Check whether it's reasonable to hide the post layer extra comm
        # since it can be overlapped
        potential_post_weights_all_gather_overlap_compute = np.maximum(
            latency_layer_exec_fwd
            - _latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act
            - np.where(
                last_layer_ckpt,
                latency_layer_swap_out_activations_with_ckpt,
                latency_layer_swap_out_activations_without_ckpt,
            ),
            0,
        )
        non_overlapped_post_weights_all_gather_latency = np.maximum(
            post_weights_all_gather_latency
            - potential_post_weights_all_gather_overlap_compute,
            0,
        )
        potential_post_grad_sync_overlap_compute_ckpt = np.maximum(
            latency_layer_exec_bwd_with_ckpt
            - latency_layer_all_gather_weights * wre_enabled
            - _latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act
            - latency_layer_swap_in_activations_with_ckpt,
            0,
        )
        potential_post_grad_sync_overlap_compute_no_ckpt = np.maximum(
            latency_layer_exec_bwd_without_ckpt
            - latency_layer_all_gather_weights * wre_enabled
            - _latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act
            - latency_layer_swap_in_activations_without_ckpt,
            0,
        )
        non_overlapped_post_grad_sync_latency = np.maximum(
            post_grad_sync_latency
            - np.where(
                last_layer_ckpt,
                potential_post_grad_sync_overlap_compute_ckpt,
                potential_post_grad_sync_overlap_compute_no_ckpt,
            ),
            0,
        )
        # post_bwd_latency -> overlapped
        overlapped_post_bwd_latency = np.maximum(
            post_bwd_latency - latency_extra_pre_bwd_not_last_micro_batch_with_ckpt, 0
        )

        latency_stable_with_post = (
            latency_stable
            + post_fwd_latency
            + overlapped_post_bwd_latency
            + np.where(
                pre_post_wre_enabled == 1,
                non_overlapped_post_weights_all_gather_latency,
                0,
            )
            + np.where(
                pre_post_gre_enabled == 1, non_overlapped_post_grad_sync_latency, 0
            )
        )
        latency_delta_with_post = (
            latency_delta
            + np.where(
                pre_post_wre_enabled == 1,
                0,
                non_overlapped_post_weights_all_gather_latency,
            )
            + np.where(
                pre_post_gre_enabled == 1, 0, non_overlapped_post_grad_sync_latency
            )
        )
        latency_with_post = (
            latency_stable_with_post
            + latency_delta_with_post / gradient_accumulation_steps
        )
        latency_stable_with_pre_and_post = (
            latency_stable
            + pre_fwd_latency
            + pre_bwd_latency
            + post_fwd_latency
            + overlapped_post_bwd_latency
            + np.where(pre_post_wre_enabled == 1, pre_weights_all_gather_latency, 0)
            + np.where(pre_post_gre_enabled == 1, pre_grad_sync_latency, 0)
            + np.where(
                pre_post_wre_enabled == 1,
                non_overlapped_post_weights_all_gather_latency,
                0,
            )
            + np.where(
                pre_post_gre_enabled == 1, non_overlapped_post_grad_sync_latency, 0
            )
        )
        latency_delta_with_pre_and_post = (
            latency_delta
            + np.where(pre_post_wre_enabled == 1, 0, pre_weights_all_gather_latency)
            + np.where(pre_post_gre_enabled == 1, 0, pre_grad_sync_latency)
            + np.where(
                pre_post_wre_enabled == 1,
                0,
                non_overlapped_post_weights_all_gather_latency,
            )
            + np.where(
                pre_post_gre_enabled == 1, 0, non_overlapped_post_grad_sync_latency
            )
        )
        latency_with_pre_and_post = (
            latency_stable_with_pre_and_post
            + latency_delta_with_pre_and_post / gradient_accumulation_steps
        )

        # calculate the memory dependent on the for-loop variables
        mem_cur_module_fwd_saved_if_ckpt_layers_equals_num_layers = (
            num_layers - 1
        ) * mem_layer_saved_with_ckpt
        mem_cur_module_fwd_saved_if_not_ckpt_layers_equals_num_layers = (
            num_ckpt_layers * mem_layer_saved_with_ckpt
            + (num_layers - num_ckpt_layers - 1) * mem_layer_saved_without_ckpt
        )
        mem_cur_module_fwd_saved_if_not_ckpt_layers_equals_num_layers = np.where(
            num_layers >= num_ckpt_layers,
            mem_cur_module_fwd_saved_if_not_ckpt_layers_equals_num_layers,
            MEM_INF,
        )
        mem_cur_module_fwd_saved = (
            mem_cur_module_fwd_saved_if_ckpt_layers_equals_num_layers
            * ckpt_layers_equals_num_layers
            + mem_cur_module_fwd_saved_if_not_ckpt_layers_equals_num_layers
            * (1 - ckpt_layers_equals_num_layers)
        )

        # calculate the module level memory dependent on the for-loop variables
        # it is mainly used for calculating the saved tensors for previous micro-batches
        mem_module_fwd_saved = (
            mem_layer_saved_with_ckpt * num_ckpt_layers
            + mem_layer_saved_without_ckpt * (num_layers - num_ckpt_layers)
        )
        mem_module_fwd_saved = np.where(
            num_layers >= num_ckpt_layers, mem_module_fwd_saved, MEM_INF
        )
        mem_module_fwd_saved_with_pre = mem_module_fwd_saved + mem_pre_layer_saved
        # mem_module_fwd_saved_with_post = mem_module_fwd_saved + mem_post_layer_saved
        # mem_module_fwd_saved_with_pre_and_post = (
        #     mem_module_fwd_saved + mem_pre_layer_saved + mem_post_layer_saved
        # )
        # consider the pre- and post- layers
        # mem_cur_module_fwd_saved_with_pre = (
        #     mem_cur_module_fwd_saved + mem_pre_layer_saved
        # )
        # this only means when the peak happens in the post
        mem_cur_module_fwd_saved_with_post = mem_module_fwd_saved
        # mem_cur_module_fwd_saved_with_pre_and_post = mem_module_fwd_saved_with_pre

        # ============================================
        # Output some extra information for debugging
        @cache
        def get_extra_info(layer_idx):
            extra_info = {
                "*****memory_factor": memory_factor[layer_idx],
                "*****mem_per_full_weight_buffer": mem_per_full_weight_buffer[
                    layer_idx
                ],
                "*****mem_per_full_grad_buffer": mem_per_full_grad_buffer[layer_idx],
                "****non_overlapped_post_weights_all_gather_latency": non_overlapped_post_weights_all_gather_latency[
                    layer_idx
                ],
                "****non_overlapped_post_grad_sync_latency": non_overlapped_post_grad_sync_latency[
                    layer_idx
                ],
                "****overlapped_post_bwd_latency": overlapped_post_bwd_latency[
                    layer_idx
                ],
                "****latency_first_layer_critical_fwd_first_micro_batch_without_ckpt": latency_first_layer_critical_fwd_first_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_last_layer_critical_fwd_first_micro_batch_without_ckpt": latency_last_layer_critical_fwd_first_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_first_layer_critical_fwd_not_first_micro_batch_without_ckpt": latency_first_layer_critical_fwd_not_first_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_last_layer_critical_fwd_not_first_micro_batch_without_ckpt": latency_last_layer_critical_fwd_not_first_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_first_layer_critical_fwd_first_micro_batch_with_ckpt": latency_first_layer_critical_fwd_first_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_last_layer_critical_fwd_first_micro_batch_with_ckpt": latency_last_layer_critical_fwd_first_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_first_layer_critical_fwd_not_first_micro_batch_with_ckpt": latency_first_layer_critical_fwd_not_first_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_last_layer_critical_fwd_not_first_micro_batch_with_ckpt": latency_last_layer_critical_fwd_not_first_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_last_layer_critical_bwd_last_micro_batch_without_ckpt": latency_last_layer_critical_bwd_last_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_first_layer_critical_bwd_last_micro_batch_without_ckpt": latency_first_layer_critical_bwd_last_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_last_layer_critical_bwd_not_last_micro_batch_without_ckpt": latency_last_layer_critical_bwd_not_last_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_first_layer_critical_bwd_not_last_micro_batch_without_ckpt": latency_first_layer_critical_bwd_not_last_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_last_layer_critical_bwd_last_micro_batch_with_ckpt": latency_last_layer_critical_bwd_last_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_first_layer_critical_bwd_last_micro_batch_with_ckpt": latency_first_layer_critical_bwd_last_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_last_layer_critical_bwd_not_last_micro_batch_with_ckpt": latency_last_layer_critical_bwd_not_last_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_first_layer_critical_bwd_not_last_micro_batch_with_ckpt": latency_first_layer_critical_bwd_not_last_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_pre_fwd_first_micro_batch_without_ckpt": latency_extra_pre_fwd_first_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_post_fwd_first_micro_batch_without_ckpt": latency_extra_post_fwd_first_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_pre_bwd_last_micro_batch_without_ckpt": latency_extra_pre_bwd_last_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_post_bwd_last_micro_batch_without_ckpt": latency_extra_post_bwd_last_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_pre_fwd_not_first_micro_batch_without_ckpt": latency_extra_pre_fwd_not_first_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_post_fwd_not_first_micro_batch_without_ckpt": latency_extra_post_fwd_not_first_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_pre_bwd_not_last_micro_batch_without_ckpt": latency_extra_pre_bwd_not_last_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_post_bwd_not_last_micro_batch_without_ckpt": latency_extra_post_bwd_not_last_micro_batch_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_pre_fwd_first_micro_batch_with_ckpt": latency_extra_pre_fwd_first_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_post_fwd_first_micro_batch_with_ckpt": latency_extra_post_fwd_first_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_pre_bwd_last_micro_batch_with_ckpt": latency_extra_pre_bwd_last_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_post_bwd_last_micro_batch_with_ckpt": latency_extra_post_bwd_last_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_pre_fwd_not_first_micro_batch_with_ckpt": latency_extra_pre_fwd_not_first_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_post_fwd_not_first_micro_batch_with_ckpt": latency_extra_post_fwd_not_first_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_pre_bwd_not_last_micro_batch_with_ckpt": latency_extra_pre_bwd_not_last_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_extra_post_bwd_not_last_micro_batch_with_ckpt": latency_extra_post_bwd_not_last_micro_batch_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_layer_critical_stable_without_ckpt": latency_layer_critical_stable_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_layer_critical_delta_without_ckpt": latency_layer_critical_delta_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_layer_critical_stable_with_ckpt": latency_layer_critical_stable_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_layer_critical_delta_with_ckpt": latency_layer_critical_delta_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_pre_fwd_and_post_bwd_and_first_layer_stable_without_ckpt": latency_pre_fwd_and_post_bwd_and_first_layer_stable_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_post_fwd_and_pre_bwd_and_last_layer_stable_without_ckpt": latency_post_fwd_and_pre_bwd_and_last_layer_stable_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_pre_fwd_and_post_bwd_and_first_layer_stable_with_ckpt": latency_pre_fwd_and_post_bwd_and_first_layer_stable_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_post_fwd_and_pre_bwd_and_last_layer_stable_with_ckpt": latency_post_fwd_and_pre_bwd_and_last_layer_stable_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_pre_fwd_and_post_bwd_and_first_layer_delta_without_ckpt": latency_pre_fwd_and_post_bwd_and_first_layer_delta_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_post_fwd_and_pre_bwd_and_last_layer_delta_without_ckpt": latency_post_fwd_and_pre_bwd_and_last_layer_delta_without_ckpt[
                    layer_idx
                ].item(),
                "****latency_pre_fwd_and_post_bwd_and_first_layer_delta_with_ckpt": latency_pre_fwd_and_post_bwd_and_first_layer_delta_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_post_fwd_and_pre_bwd_and_last_layer_delta_with_ckpt": latency_post_fwd_and_pre_bwd_and_last_layer_delta_with_ckpt[
                    layer_idx
                ].item(),
                "****latency_stable": latency_stable[layer_idx].item(),
                "****latency_delta": latency_delta[layer_idx].item(),
                "****latency_stable_with_pre": latency_stable_with_pre[
                    layer_idx
                ].item(),
                "****latency_delta_with_pre": latency_delta_with_pre[layer_idx].item(),
                "****latency_stable_with_post": latency_stable_with_post[
                    layer_idx
                ].item(),
                "****latency_delta_with_post": latency_delta_with_post[
                    layer_idx
                ].item(),
                "****latency_stable_with_pre_and_post": latency_stable_with_pre_and_post[
                    layer_idx
                ].item(),
                "****latency_delta_with_pre_and_post": latency_delta_with_pre_and_post[
                    layer_idx
                ].item(),
                "_latency_pre_fwd_latency": pre_fwd_latency.item(),
                "_latency_pre_bwd_latency": pre_bwd_latency.item(),
                "_latency_pre_weights_all_gather_latency": pre_weights_all_gather_latency.item(),
                "_latency_pre_weights_all_gather_latency_amortized": pre_weights_all_gather_latency_amortized.item(),
                "_latency_pre_grad_sync_latency": pre_grad_sync_latency.item(),
                "_latency_pre_grad_sync_amortized": pre_grad_sync_latency_amortized,
                "_latency_post_fwd_latency": post_fwd_latency.item(),
                "_latency_post_bwd_latency": post_bwd_latency.item(),
                "_latentcy_post_weights_all_gather_latency_amortized": post_weights_all_gather_latency_amortized.item(),
                "_latency_post_grad_sync_amortized": post_grad_sync_latency_amortized,
                "_latency_post_grad_sync_latency": post_grad_sync_latency.item(),
                "_latency_post_weights_all_gather_latency": post_weights_all_gather_latency.item(),
                "_latency_layer_exec_fwd": latency_layer_exec_fwd[layer_idx],
                "_latency_layer_exec_bwd_without_ckpt": latency_layer_exec_bwd_without_ckpt[
                    layer_idx
                ],
                "_latency_layer_exec_bwd_with_ckpt": latency_layer_exec_bwd_with_ckpt[
                    layer_idx
                ],
                "_latency_layer_gpu_to_cpu_fwd_first_micro_batch_exclude_act": _latency_layer_gpu_to_cpu_fwd_first_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_cpu_to_gpu_fwd_first_micro_batch_exclude_act": _latency_layer_cpu_to_gpu_fwd_first_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_gpu_gpu_fwd_first_micro_batch_exclude_act": _latency_layer_gpu_gpu_fwd_first_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act": _latency_layer_gpu_to_cpu_fwd_not_first_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_cpu_to_gpu_fwd_not_first_micro_batch_exclude_act": _latency_layer_cpu_to_gpu_fwd_not_first_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_gpu_gpu_fwd_not_first_micro_batch_exclude_act": _latency_layer_gpu_gpu_fwd_not_first_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_gpu_to_cpu_bwd_last_micro_batch_exclude_act": _latency_layer_gpu_to_cpu_bwd_last_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_cpu_to_gpu_bwd_last_micro_batch_exclude_act": _latency_layer_cpu_to_gpu_bwd_last_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_gpu_gpu_bwd_last_micro_batch_exclude_act": _latency_layer_gpu_gpu_bwd_last_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_gpu_to_cpu_bwd_not_last_micro_batch_exclude_act": _latency_layer_gpu_to_cpu_bwd_not_last_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act": _latency_layer_cpu_to_gpu_bwd_not_last_micro_batch_exclude_act[
                    layer_idx
                ],
                "_latency_layer_gpu_gpu_bwd_not_last_micro_batch_exclude_act": _latency_layer_gpu_gpu_bwd_not_last_micro_batch_exclude_act[
                    layer_idx
                ],
                "**_latency_layer_critical_fwd_first_micro_batch_without_ckpt": latency_layer_critical_fwd_first_micro_batch_without_ckpt[
                    layer_idx
                ],
                "*_latency_layer_critical_fwd_first_micro_batch_with_ckpt": latency_layer_critical_fwd_first_micro_batch_with_ckpt[
                    layer_idx
                ],
                "**_latency_layer_critical_fwd_not_first_micro_batch_without_ckpt": latency_layer_critical_fwd_not_first_micro_batch_without_ckpt[
                    layer_idx
                ],
                "*_latency_layer_critical_fwd_not_first_micro_batch_with_ckpt": latency_layer_critical_fwd_not_first_micro_batch_with_ckpt[
                    layer_idx
                ],
                "**_latency_layer_critical_bwd_last_micro_batch_without_ckpt": latency_layer_critical_bwd_last_micro_batch_without_ckpt[
                    layer_idx
                ],
                "*_latency_layer_critical_bwd_last_micro_batch_with_ckpt": latency_layer_critical_bwd_last_micro_batch_with_ckpt[
                    layer_idx
                ],
                "**_latency_layer_critical_bwd_not_last_micro_batch_without_ckpt": latency_layer_critical_bwd_not_last_micro_batch_without_ckpt[
                    layer_idx
                ],
                "*_latency_layer_critical_bwd_not_last_micro_batch_with_ckpt": latency_layer_critical_bwd_not_last_micro_batch_with_ckpt[
                    layer_idx
                ],
                "_latency_layer_swap_out_weights": latency_layer_swap_out_weights[
                    layer_idx
                ],
                "_latency_layer_swap_in_weights": latency_layer_swap_in_weights[
                    layer_idx
                ],
                "_latency_layer_all_gather_weights": latency_layer_all_gather_weights[
                    layer_idx
                ],
                "_latency_layer_swap_in_grads": latency_layer_swap_in_grads[layer_idx],
                "_latency_layer_swap_in_activations_with_ckpt": latency_layer_swap_in_activations_with_ckpt[
                    layer_idx
                ],
                "_latency_layer_swap_in_activations_without_ckpt": latency_layer_swap_in_activations_without_ckpt[
                    layer_idx
                ],
                "_latency_layer_swap_out_grads": latency_layer_swap_out_grads[
                    layer_idx
                ],
                "_latency_layer_swap_out_activations_with_ckpt": latency_layer_swap_out_activations_with_ckpt[
                    layer_idx
                ],
                "_latency_layer_swap_out_activations_without_ckpt": latency_layer_swap_out_activations_without_ckpt[
                    layer_idx
                ],
                "_latency_layer_reduce_scatter_grads": latency_layer_reduce_scatter_grads[
                    layer_idx
                ],
                "_latency_layer_bwd_all_gather_saved_with_ckpt": latency_layer_bwd_all_gather_saved_with_ckpt[
                    layer_idx
                ],
                # "_latency_fwd": fwd_latency[layer_idx],
                # "_latency_bwd": bwd_latency[layer_idx],
                # "_latency_fwd_with_pre_and_post": fwd_latency[layer_idx]
                # + pre_fwd_latency.item()
                # + post_fwd_latency.item(),
                # "_latency_fwd_with_pre": fwd_latency[layer_idx]
                # + pre_fwd_latency.item(),
                # "_latency_fwd_with_post": fwd_latency[layer_idx]
                # + post_fwd_latency.item(),
                # "_latency_bwd_with_pre_and_post": bwd_latency[layer_idx]
                # + pre_bwd_latency.item()
                # + post_bwd_latency.item(),
                # "_latency_bwd_with_pre": bwd_latency[layer_idx]
                # + pre_bwd_latency.item(),
                # "_latency_bwd_with_post": bwd_latency[layer_idx]
                # + post_bwd_latency.item(),
                # "_latency_layer_critical_fwd_with_ckpt": latency_layer_critical_fwd_with_ckpt[
                #     layer_idx
                # ],
                # "_latency_layer_critical_fwd_without_ckpt": latency_layer_critical_fwd_without_ckpt[
                #     layer_idx
                # ],
                # "_latency_layer_critical_bwd_with_ckpt": latency_layer_critical_bwd_with_ckpt[
                #     layer_idx
                # ],
                # "_latency_layer_critical_bwd_without_ckpt": latency_layer_critical_bwd_without_ckpt[
                #     layer_idx
                # ],
                "_latency_p2p_output": latency_p2p_output[layer_idx],
                # "_mem_extra_buffer": mem_extra_buffer[layer_idx],
                "_mem_layer_full_weights": mem_layer_full_weights[layer_idx],
                "_mem_layer_full_grads": mem_layer_full_grads[layer_idx],
                "_mem_layer_full_opts": mem_layer_full_opts[layer_idx],
                "_mem_layer_partial_weights": mem_layer_partial_weights[layer_idx],
                "_mem_layer_partial_grads": mem_layer_partial_grads[layer_idx],
                "_mem_layer_partial_opts": mem_layer_partial_opts[layer_idx],
                "_mem_layer_saved_with_ckpt": mem_layer_saved_with_ckpt[layer_idx],
                "_mem_layer_saved_without_ckpt": mem_layer_saved_without_ckpt[
                    layer_idx
                ],
                "_mem_raw_layer_saved_with_ckpt": _raw_mem_layer_saved_with_ckpt[
                    layer_idx
                ],
                "_mem_raw_layer_saved_without_ckpt": _raw_mem_layer_saved_without_ckpt[
                    layer_idx
                ],
                "_mem_layer_fwd_peak_with_ckpt": mem_layer_fwd_peak_with_ckpt[
                    layer_idx
                ],
                "_mem_layer_fwd_peak_without_ckpt": mem_layer_fwd_peak_without_ckpt[
                    layer_idx
                ],
                "_mem_layer_bwd_peak": mem_layer_bwd_peak[layer_idx],
                "_mem_layer_output": mem_layer_output[layer_idx],
                "_mem_module_fwd_saved": mem_module_fwd_saved[layer_idx],
                "_mem_pre_layer_full_weights": mem_pre_layer_full_weights.item(),
                "_mem_pre_layer_full_grads": mem_pre_layer_full_grads.item(),
                "_mem_pre_layer_full_opts": mem_pre_layer_full_opts.item(),
                "_mem_pre_layer_saved_without_ckpt": mem_pre_layer_saved.item(),
                "_mem_pre_layer_states": mem_pre_layer_states.item(),
                "_mem_post_layer_full_weights": mem_post_layer_full_weights.item(),
                "_mem_post_layer_full_grads": mem_post_layer_full_grads.item(),
                "_mem_post_layer_full_opts": mem_post_layer_full_opts.item(),
                "_mem_post_layer_saved_without_ckpt": mem_post_layer_saved.item(),
                "_mem_post_layer_states": mem_post_layer_states.item(),
                "_mem_post_layer_fwd_peak": mem_post_layer_fwd_peak.item(),
                "_mem_post_layer_bwd_peak": mem_post_layer_bwd_peak.item(),
                "_mem_post_layer_output": mem_post_layer_output.item(),
                "_mem_module_fwd_states": mem_module_fwd_states[layer_idx],
                "_mem_module_bwd_states": mem_module_bwd_states[layer_idx],
            }
            return extra_info

        # ============================================

        for pre_saved_micro_batches in pre_saved_micro_batches_candidates:
            # Consider the memory related to the pre_saved_micro_batches
            mem_pipe_fwd_saved = (
                # Pre-stages
                pre_saved_micro_batches * mem_module_fwd_saved
                + mem_cur_module_fwd_saved
            )
            ckpt_layers_larger_equal_num_layers_minus_1 = num_ckpt_layers >= (
                num_layers - 1
            )
            mem_ckpt_delta_for_peak = (
                mem_layer_saved_with_ckpt_delta_for_peak
                * ckpt_layers_larger_equal_num_layers_minus_1
            ) + (
                mem_layer_saved_without_ckpt_delta_for_peak
                * (1 - ckpt_layers_larger_equal_num_layers_minus_1)
            )
            mem_pipe_fwd_peak_no_pre_and_post = (
                mem_module_fwd_states
                + mem_pipe_fwd_saved
                + mem_ckpt_delta_for_peak
                + mem_layer_fwd_peak_without_ckpt
                + mem_layer_output * pre_saved_micro_batches * 2
                + mem_global_constant_buffer
            )
            mem_pipe_bwd_peak_no_pre_and_post = (
                mem_module_bwd_states
                + mem_pipe_fwd_saved
                + mem_ckpt_delta_for_peak
                + mem_layer_bwd_peak
                + mem_layer_output * pre_saved_micro_batches * 2
                + mem_layer_output * 2
                + mem_global_constant_buffer
            )

            if not get_best_solution:
                results.update(
                    {
                        (
                            pre_saved_micro_batches,
                            num_layers_candidates[i],
                            num_ckpt_layers,
                        ): {
                            "latency": latency[i] + latency_p2p_output[i],
                            "mem_fwd_peak": mem_pipe_fwd_peak_no_pre_and_post[i],
                            "mem_bwd_peak": mem_pipe_bwd_peak_no_pre_and_post[i],
                            **get_extra_info(i),
                        }
                        for i in range(len(num_layers_candidates))
                    }
                )
            else:
                mask = (
                    np.maximum(
                        mem_pipe_fwd_peak_no_pre_and_post,
                        mem_pipe_bwd_peak_no_pre_and_post,
                    )
                    * (memory_factor_for_pp + 0.01)
                    <= config.hardware.memory_capacity * GB / MB
                )
                indices, _, _ = sample_pareto_frontier(
                    costs_x=latency_stable,
                    costs_y=latency_delta / gradient_accumulation_steps,
                    mask=mask,
                    sample_size=sample_size,
                )
                selected_costs_stable = latency_stable[np.arange(n)[:, None], indices]
                selected_costs_delta = latency_delta[np.arange(n)[:, None], indices]
                selected_solution = stage_strategies[indices]
                selected_costs_stable[indices == -1] = np.inf
                selected_costs_delta[indices == -1] = np.inf
                selected_solution[indices == -1] = np.inf
                # Update the costs and solutions
                costs_stable[
                    pre_saved_micro_batches, valid_layer_mask, num_ckpt_layers
                ] = selected_costs_stable[valid_layer_mask]
                costs_delta[
                    pre_saved_micro_batches, valid_layer_mask, num_ckpt_layers
                ] = selected_costs_delta[valid_layer_mask]
                solutions[
                    pre_saved_micro_batches, valid_layer_mask, num_ckpt_layers
                ] = selected_solution[valid_layer_mask]

                # ============================================
                # # Previous implementation
                # penalty = np.where(
                #     np.maximum(
                #         mem_pipe_fwd_peak_no_pre_and_post,
                #         mem_pipe_bwd_peak_no_pre_and_post,
                #     )
                #     * memory_factor
                #     > config.hardware.memory_capacity * GB / MB,
                #     np.inf,
                #     0,
                # )
                # objective = latency + latency_p2p_output + penalty
                # best_objective = np.min(objective, axis=-1)
                # best_solution_index = np.argmin(objective, axis=-1)
                # best_solutions = stage_strategies[best_solution_index]
                # best_solutions[best_objective == np.inf] = np.inf
                # # TODO(zhanda): when num_ckpt_layers > num_layers, costs should be inf.
                # # Currently, this is not guaranteed. But it is not a problem for now.
                # # this can be solved later because we won't access the costs when
                # # num_ckpt_layers > num_layers
                # costs[pre_saved_micro_batches, valid_layer_mask, num_ckpt_layers] = (
                #     best_objective[valid_layer_mask]
                # )
                # solutions[
                #     pre_saved_micro_batches, valid_layer_mask, num_ckpt_layers
                # ] = best_solutions[valid_layer_mask]
                # ============================================

            # consider the effect of the pre- layers
            mem_pipe_fwd_peak = (
                pre_saved_micro_batches * mem_pre_layer_saved
                + mem_pipe_fwd_peak_no_pre_and_post
                + mem_pre_layer_states_and_saved
            )
            mem_pipe_bwd_peak = (
                pre_saved_micro_batches * mem_pre_layer_saved
                + mem_pipe_bwd_peak_no_pre_and_post
                + mem_pre_layer_states_and_saved
            )
            if not get_best_solution:
                results_with_pre.update(
                    {
                        (
                            pre_saved_micro_batches,
                            num_layers_candidates[i],
                            num_ckpt_layers,
                        ): {
                            "latency": latency_with_pre[i] + latency_p2p_output[i],
                            "mem_fwd_peak": mem_pipe_fwd_peak[i],
                            "mem_bwd_peak": mem_pipe_bwd_peak[i],
                            **get_extra_info(i),
                        }
                        for i in range(len(num_layers_candidates))
                    }
                )
            else:
                mask = (
                    np.maximum(mem_pipe_fwd_peak, mem_pipe_bwd_peak)
                    * memory_factor_for_pp
                    <= config.hardware.memory_capacity * GB / MB
                )
                indices, _, _ = sample_pareto_frontier(
                    costs_x=latency_stable_with_pre,
                    costs_y=latency_delta_with_pre / gradient_accumulation_steps,
                    mask=mask,
                    sample_size=sample_size,
                )
                selected_costs_stable = latency_stable_with_pre[
                    np.arange(n)[:, None], indices
                ]
                selected_costs_delta = latency_delta_with_pre[
                    np.arange(n)[:, None], indices
                ]
                selected_solution = stage_strategies[indices]
                selected_costs_stable[indices == -1] = np.inf
                selected_costs_delta[indices == -1] = np.inf
                selected_solution[indices == -1] = np.inf
                # Update the costs and solutions
                costs_stable_with_pre[
                    pre_saved_micro_batches, valid_layer_mask, num_ckpt_layers
                ] = selected_costs_stable[valid_layer_mask]
                costs_delta_with_pre[
                    pre_saved_micro_batches, valid_layer_mask, num_ckpt_layers
                ] = selected_costs_delta[valid_layer_mask]
                solutions_with_pre[
                    pre_saved_micro_batches, valid_layer_mask, num_ckpt_layers
                ] = selected_solution[valid_layer_mask]

                # ============================================
                # # Previous implementation
                # penalty = np.where(
                #     np.maximum(mem_pipe_fwd_peak, mem_pipe_bwd_peak) * memory_factor
                #     > config.hardware.memory_capacity * GB / MB,
                #     np.inf,
                #     0,
                # )
                # objective = latency_with_pre + latency_p2p_output + penalty
                # best_objective = np.min(objective, axis=-1)
                # best_solution_index = np.argmin(objective, axis=-1)
                # best_solutions = stage_strategies[best_solution_index]
                # best_solutions[best_objective == np.inf] = np.inf
                # costs_with_pre[
                #     pre_saved_micro_batches, valid_layer_mask, num_ckpt_layers
                # ] = best_objective[valid_layer_mask]
                # solutions_with_pre[
                #     pre_saved_micro_batches, valid_layer_mask, num_ckpt_layers
                # ] = best_solutions[valid_layer_mask]
                # ============================================

            if pre_saved_micro_batches == 0:
                # consider the effect of the post- layers
                mem_pipe_fwd_saved_with_post = mem_cur_module_fwd_saved_with_post
                mem_pipe_fwd_peak_in_post_layer = (
                    mem_module_fwd_states
                    + mem_post_layer_states
                    + mem_pipe_fwd_saved_with_post
                    + mem_post_layer_fwd_peak
                    + mem_global_constant_buffer
                )
                mem_pipe_bwd_peak_in_post_layer = (
                    mem_module_bwd_states
                    + mem_post_layer_states
                    + mem_pipe_fwd_saved_with_post
                    + mem_post_layer_bwd_peak
                    # Delta for the second last layer's activation if it's swapped out
                    # and gradient checkpointing
                    + (_raw_mem_layer_saved_without_ckpt - mem_layer_saved_without_ckpt)
                    * ckpt_layers_equals_zero
                    + mem_global_constant_buffer
                )
                mem_pipe_fwd_peak = np.maximum(
                    mem_pipe_fwd_peak_in_post_layer,
                    mem_pipe_fwd_peak_no_pre_and_post + mem_post_layer_states,
                )
                mem_pipe_bwd_peak = (
                    np.maximum(
                        mem_pipe_bwd_peak_in_post_layer,
                        mem_pipe_bwd_peak_no_pre_and_post
                        - mem_layer_output
                        + mem_post_layer_states,
                    )
                    + mem_post_layer_output
                )
                if not get_best_solution:
                    results_with_post.update(
                        {
                            (
                                pre_saved_micro_batches,
                                num_layers_candidates[i],
                                num_ckpt_layers,
                            ): {
                                "latency": latency_with_post[i] + latency_p2p_output[i],
                                "mem_fwd_peak": mem_pipe_fwd_peak[i],
                                "mem_bwd_peak": mem_pipe_bwd_peak[i],
                                **get_extra_info(i),
                            }
                            for i in range(len(num_layers_candidates))
                        }
                    )
                else:
                    mask = (
                        np.maximum(mem_pipe_fwd_peak, mem_pipe_bwd_peak)
                        * memory_factor_for_pp
                        <= config.hardware.memory_capacity * GB / MB
                    )
                    indices, _, _ = sample_pareto_frontier(
                        costs_x=latency_stable_with_post,
                        costs_y=latency_delta_with_post / gradient_accumulation_steps,
                        mask=mask,
                        sample_size=sample_size,
                    )
                    selected_costs_stable = latency_stable_with_post[
                        np.arange(n)[:, None], indices
                    ]
                    selected_costs_delta = latency_delta_with_post[
                        np.arange(n)[:, None], indices
                    ]
                    selected_solution = stage_strategies[indices]
                    selected_costs_stable[indices == -1] = np.inf
                    selected_costs_delta[indices == -1] = np.inf
                    selected_solution[indices == -1] = np.inf
                    # Update the costs and solutions
                    # ==========================================================================================
                    # TODO(zhanda): Improve the memory prediction when num_ckpt_layers == num_layers - 1
                    valid_layer_mask = np.logical_and(
                        valid_layer_mask, num_ckpt_layers != num_layers.flatten() - 1
                    )
                    # ==========================================================================================
                    costs_stable_with_post[valid_layer_mask, num_ckpt_layers] = (
                        selected_costs_stable[valid_layer_mask]
                    )
                    costs_delta_with_post[valid_layer_mask, num_ckpt_layers] = (
                        selected_costs_delta[valid_layer_mask]
                    )
                    solutions_with_post[valid_layer_mask, num_ckpt_layers] = (
                        selected_solution[valid_layer_mask]
                    )

                    # ============================================
                    # # Previous implementation
                    # penalty = np.where(
                    #     np.maximum(mem_pipe_fwd_peak, mem_pipe_bwd_peak) * memory_factor
                    #     > config.hardware.memory_capacity * GB / MB,
                    #     np.inf,
                    #     0,
                    # )
                    # objective = latency_with_post + latency_p2p_output + penalty
                    # best_objective = np.min(objective, axis=-1)
                    # best_solution_index = np.argmin(objective, axis=-1)
                    # best_solutions = stage_strategies[best_solution_index]
                    # best_solutions[best_objective == np.inf] = np.inf
                    # costs_with_post[valid_layer_mask, num_ckpt_layers] = best_objective[
                    #     valid_layer_mask
                    # ]
                    # solutions_with_post[valid_layer_mask, num_ckpt_layers] = (
                    #     best_solutions[valid_layer_mask]
                    # )
                    # ============================================

                # consider the effect of the pre- and post- layers
                mem_pipe_fwd_peak = (
                    mem_pipe_fwd_peak + mem_pre_layer_states + mem_pre_layer_saved
                )
                mem_pipe_bwd_peak = (
                    mem_pipe_bwd_peak + mem_pre_layer_states + mem_pre_layer_saved
                )
                if not get_best_solution:
                    reulst_with_pre_and_post.update(
                        {
                            (
                                pre_saved_micro_batches,
                                num_layers_candidates[i],
                                num_ckpt_layers,
                            ): {
                                "latency": latency_with_pre_and_post[i],
                                "mem_fwd_peak": mem_pipe_fwd_peak[i],
                                "mem_bwd_peak": mem_pipe_bwd_peak[i],
                                **get_extra_info(i),
                            }
                            for i in range(len(num_layers_candidates))
                        }
                    )
                else:
                    mask = (
                        np.maximum(mem_pipe_fwd_peak, mem_pipe_bwd_peak) * memory_factor
                        <= config.hardware.memory_capacity * GB / MB
                    )
                    indices, _, _ = sample_pareto_frontier(
                        costs_x=latency_stable_with_pre_and_post,
                        costs_y=latency_delta_with_pre_and_post
                        / gradient_accumulation_steps,
                        mask=mask,
                        sample_size=sample_size,
                    )
                    selected_costs_stable = latency_stable_with_pre_and_post[
                        np.arange(n)[:, None], indices
                    ]
                    selected_costs_delta = latency_delta_with_pre_and_post[
                        np.arange(n)[:, None], indices
                    ]
                    selected_solution = stage_strategies[indices]
                    selected_costs_stable[indices == -1] = np.inf
                    selected_costs_delta[indices == -1] = np.inf
                    selected_solution[indices == -1] = np.inf
                    # Update the costs and solutions
                    # TODO(zhanda): Improve the memory prediction when num_ckpt_layers == num_layers - 1
                    if num_ckpt_layers != c - 1:
                        costs_stable_no_pp[num_ckpt_layers] = selected_costs_stable[-1]
                        costs_delta_no_pp[num_ckpt_layers] = selected_costs_delta[-1]
                        solutions_no_pp[num_ckpt_layers] = selected_solution[-1]

                    # ============================================
                    # # Previous implementation
                    # penalty = np.where(
                    #     np.maximum(mem_pipe_fwd_peak, mem_pipe_bwd_peak) * memory_factor
                    #     > config.hardware.memory_capacity * GB / MB,
                    #     np.inf,
                    #     0,
                    # )
                    # objective = latency_with_pre_and_post + penalty
                    # best_objective = np.min(objective, axis=-1)
                    # best_solution_index = np.argmin(objective, axis=-1)
                    # best_solutions = stage_strategies[best_solution_index]
                    # best_solutions[best_objective == np.inf] = np.inf
                    # costs_no_pp[num_ckpt_layers] = best_objective[-1]
                    # solutions_no_pp[num_ckpt_layers] = best_solutions[-1]
                    # ============================================

                    # Debugging example
                    # if num_ckpt_layers == 32 and gradient_accumulation_steps == 2:
                    #     print(mem_pipe_fwd_peak.shape)
                    #     print(
                    #         f"[num_nodes={num_nodes}, num_gpus_per_node={num_gpus_per_node}] "
                    #         f"costs_no_pp: {costs_no_pp[num_ckpt_layers]}, "
                    #         f"solutions_no_pp: {solutions_no_pp[num_ckpt_layers]} "
                    #         f"mem_pipe_fwd_peak: {mem_pipe_fwd_peak[-1, best_solution_index[-1]]}, "
                    #         f"mem_pipe_bwd_peak: {mem_pipe_bwd_peak[-1, best_solution_index[-1]]}, "
                    #     )

    if not get_best_solution:
        return results, results_with_pre, results_with_post, reulst_with_pre_and_post
    else:
        return (
            (costs_stable, costs_delta, solutions),
            (costs_stable_with_pre, costs_delta_with_pre, solutions_with_pre),
            (costs_stable_with_post, costs_delta_with_post, solutions_with_post),
            (costs_stable_no_pp, costs_delta_no_pp, solutions_no_pp),
        )
