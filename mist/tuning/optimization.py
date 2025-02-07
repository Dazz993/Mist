import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, cache
from time import perf_counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional, TYPE_CHECKING, Any


import numpy as np
import torch

from mist.config import MistConfig
from mist.analyzer.info import LayerInfo
from mist.analyzer.strategy import (
    ModelStrategy,
    LayerStrategy,
    PhaseStrategy,
    create_model_strategy_from_layer_strategies,
)
from mist.analyzer.model_analyzer import ModelAnalyzer
from mist.logger import get_logger
from mist.utils.pipeline_parallel import calculate_num_warmup_and_1f1b_phases
from mist.tuning.batched_model_optim_prob import ModelGranularityOptimProb
from mist.analyzer.batched_module_analyzer import batched_tune_best_latency_for_stage

logger = get_logger(__name__)

POWER_OF_TWO = [2**i for i in range(0, 15)]


def _get_device_mesh_candidates(
    num_nodes, num_gpus_per_node, contiguous_inter_node: bool = True
):
    device_mesh_candidates = []
    for m in range(1, num_gpus_per_node + 1):
        if m in POWER_OF_TWO:
            device_mesh_candidates.append((1, m))
    if contiguous_inter_node:
        for n in range(2, num_nodes + 1):
            device_mesh_candidates.append((n, num_gpus_per_node))
    else:
        for n in range(2, num_nodes + 1):
            if n in POWER_OF_TWO:
                device_mesh_candidates.append((n, num_gpus_per_node))
    # Assert no duplicates
    assert len(device_mesh_candidates) == len(set(device_mesh_candidates)), (
        f"Duplicate device mesh candidates: {device_mesh_candidates}"
    )
    return device_mesh_candidates


def _get_grad_accumu_steps_candidates(global_batch_size):
    return [i for i in POWER_OF_TWO if global_batch_size % i == 0]


def _uniform_layer_partition(num_layers: int, num_stages: int) -> List[int]:
    """Uniformly partition the layers into stages."""
    num_layers_per_stage = num_layers // num_stages
    layer_partition = [num_layers_per_stage] * num_stages
    # Add layers to the last few stages
    for i in range(num_layers % num_stages):
        layer_partition[-i - 1] += 1
    return layer_partition


def _uniform_gpu_partition_for_pp(num_stages, num_nodes, num_gpus_per_node):
    if num_nodes > num_stages:
        assert num_nodes % num_stages == 0
        num_nodes = num_nodes // num_stages
        num_gpus_per_node = num_gpus_per_node
    else:
        num_nodes = 1
        assert num_stages % num_nodes == 0
        assert num_gpus_per_node % (num_stages // num_nodes) == 0
        num_gpus_per_node = num_gpus_per_node // (num_stages // num_nodes)

    return num_nodes, num_gpus_per_node


def build_and_tune_optimization_problem(
    block_layer_info: LayerInfo,
    config: MistConfig,
    pre_layer_info: Optional[LayerInfo] = None,
    post_layer_info: Optional[LayerInfo] = None,
):
    grad_accumu_steps_tuning_enabled = config.grad_accumu_steps_tuning_enabled
    num_stages_candidates = config.num_stages_candidates_if_auto_tuning_disabled
    layer_partitioning_tuning_enabled = config.layer_partitioning_tuning_enabled
    device_assignment_tuning_enabled = config.device_assignment_tuning_enabled
    if not layer_partitioning_tuning_enabled and device_assignment_tuning_enabled:
        raise ValueError(
            "Cannot tune device assignment without tuning layer partitioning."
        )

    if grad_accumu_steps_tuning_enabled:
        gradient_accumulation_steps_candidates = [
            i
            for i in range(1, config.global_batch_size + 1)
            if config.global_batch_size % i == 0
        ]
    else:
        gradient_accumulation_steps_candidates = (
            config.grad_accumu_steps_tuning_candidates_if_auto_tuning_disabled
        )

    results = []
    # Tune layer partition and device assignment
    if layer_partitioning_tuning_enabled and device_assignment_tuning_enabled:
        for gradient_accumulation_steps in gradient_accumulation_steps_candidates:
            result = build_and_tune_optimization_problem_with_pp_tuned(
                block_layer_info=block_layer_info,
                gradient_accumulation_steps=gradient_accumulation_steps,
                config=config,
                pre_layer_info=pre_layer_info,
                post_layer_info=post_layer_info,
            )
            results.append(result)

    # Tune layer partition but not device assignment
    elif layer_partitioning_tuning_enabled and not device_assignment_tuning_enabled:
        for num_stages in num_stages_candidates:
            (
                num_nodes_per_stage,
                num_gpus_per_node_per_stage,
            ) = _uniform_gpu_partition_for_pp(
                num_stages, config.num_nodes, config.num_gpus_per_node
            )
            for gradient_accumulation_steps in gradient_accumulation_steps_candidates:
                result = build_and_tune_optimization_problem_with_layer_partition_tuned(
                    block_layer_info=block_layer_info,
                    num_stages=config.num_stages_if_tuning_disabled,
                    num_nodes_per_stage=num_nodes_per_stage,
                    num_gpus_per_node_per_stage=num_gpus_per_node_per_stage,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    config=config,
                    pre_layer_info=pre_layer_info,
                    post_layer_info=post_layer_info,
                )
                results.append(result)

    # Tune neither layer partition nor device assignment
    else:
        for num_stages in num_stages_candidates:
            block_layer_partition = _uniform_layer_partition(
                config.num_layers, num_stages=num_stages
            )
            (
                num_nodes_per_stage,
                num_gpus_per_node_per_stage,
            ) = _uniform_gpu_partition_for_pp(
                num_stages, config.num_nodes, config.num_gpus_per_node
            )
            for gradient_accumulation_steps in gradient_accumulation_steps_candidates:
                result = build_and_tune_optimization_problem_with_pp_fixed(
                    block_layer_info=block_layer_info,
                    num_stages=config.num_stages_if_tuning_disabled,
                    block_layer_partition=block_layer_partition,
                    num_nodes_per_stage=num_nodes_per_stage,
                    num_gpus_per_node_per_stage=num_gpus_per_node_per_stage,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    config=config,
                    pre_layer_info=pre_layer_info,
                    post_layer_info=post_layer_info,
                )
                results.append(result)

    return results


def build_and_tune_optimization_problem_with_pp_tuned(
    block_layer_info: LayerInfo,
    gradient_accumulation_steps_candidates: List[int],
    config: MistConfig,
    pre_layer_info: Optional[LayerInfo] = None,
    post_layer_info: Optional[LayerInfo] = None,
):
    """Only consider num stages > 1. Non-pp should be considered outside."""


def calculate_stage_latencies(
    block_layer_info: LayerInfo,
    num_stages: int,
    num_layers: int,
    num_nodes_per_stage: int,
    num_gpus_per_node_per_stage: int,
    gradient_accumulation_steps: int,
    config: MistConfig,
    pre_layer_info: Optional[LayerInfo] = None,
    post_layer_info: Optional[LayerInfo] = None,
):
    # samples for pre-layer and post-layer: (num_layers, num_layers + 1)
    # samples for block-layer: (num_stages, num_layers, num_layers + 1)

    is_first_stage = pre_layer_info is not None
    is_last_stage = post_layer_info is not None

    if is_first_stage or is_last_stage:
        # For pre/post -layer:
        # features_np: (num_stages, stage_idx, num_layers, num_ckpt_layers)
        #                                      ^^^^^^^^^^  ^^^^^^^^^^^^^^^
        #                                      [1, num_layers], [0, num_layers]
        stage_idx = 0 if is_first_stage else num_stages - 1
        pre_post_features_np = np.zeros((num_layers * (num_layers + 1), 4), dtype=float)
        pre_post_features_np[:, 0] = num_stages
        pre_post_features_np[:, 1] = stage_idx
        pre_post_features_np[:, 2] = np.repeat(
            np.arange(1, num_layers + 1), num_layers + 1
        )
        pre_post_features_np[:, 3] = np.tile(np.arange(num_layers + 1), num_layers)
        # TODO(zhanda): Drop invalid samples in the callee
        # pre_features_np = pre_features_np[pre_features_np[:, 3] <= pre_features_np[:, 2]]
        pre_stage_latencies, pre_stage_strategies = batched_tune_best_latency_for_stage(
            block_layer_info=block_layer_info,
            features_np=pre_post_features_np,
            num_nodes=num_nodes_per_stage,
            num_gpus_per_node=num_gpus_per_node_per_stage,
            gradient_accumulation_steps=gradient_accumulation_steps,
            config=config,
            pre_layer_info=pre_layer_info,
            post_layer_info=None,
        )
        pre_stage_latencies = pre_stage_latencies.reshape((num_layers, num_layers + 1))
        pre_stage_strategies = pre_stage_strategies.reshape(
            (num_layers, num_layers + 1, -1)
        )
        return pre_stage_latencies, pre_stage_strategies

    else:
        # * For block layers:
        # features_np: (num_stages, stage_idx, num_layers, num_ckpt_layers)
        #                           ^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^^^^
        #                           [1, num_stages - 1], [1, num_layers], [0, num_layers]
        block_features_np = np.zeros(
            ((num_stages - 1) * num_layers * (num_layers + 1), 4), dtype=float
        )
        block_features_np[:, 0] = num_stages
        block_features_np[:, 1] = np.repeat(
            np.arange(1, num_stages), num_layers * (num_layers + 1)
        )
        block_features_np[:, 2] = np.tile(
            np.repeat(np.arange(1, num_layers + 1), num_layers + 1), num_stages - 1
        )
        block_features_np[:, 3] = np.tile(
            np.arange(num_layers + 1), (num_stages - 1) * num_layers
        )
        (
            block_stage_latencies,
            block_stage_strategies,
        ) = batched_tune_best_latency_for_stage(
            block_layer_info=block_layer_info,
            features_np=block_features_np,
            num_nodes=num_nodes_per_stage,
            num_gpus_per_node=num_gpus_per_node_per_stage,
            gradient_accumulation_steps=gradient_accumulation_steps,
            config=config,
            pre_layer_info=None,
            post_layer_info=None,
        )
        block_stage_latencies = block_stage_latencies.reshape(
            (num_stages - 1, num_layers, num_layers + 1)
        )
        block_stage_strategies = block_stage_strategies.reshape(
            (num_stages - 1, num_layers, num_layers + 1, -1)
        )

        return block_stage_latencies, block_stage_strategies


def build_and_tune_optimization_problem_with_layer_partition_tuned(
    block_layer_info: LayerInfo,
    num_stages: int,
    num_nodes_per_stage: int,
    num_gpus_per_node_per_stage: int,
    gradient_accumulation_steps: int,
    config: MistConfig,
    pre_layer_info: Optional[LayerInfo] = None,
    post_layer_info: Optional[LayerInfo] = None,
):
    """Given the number of stages and device assignment, build and tune the optimization problem.

    The parameters needed to be tuned are:
    - the number of layers for each stage
    - the number of ckpt layers for each stage
    - the strategies for each stages

    DP Form:
    F(s, k, c) means the minimum latency for the layers(L_1, ..., L_k) with c ckpt layers in s stages
    F(s, k, c) = min_{1 <= i <= k, 0 <= j <= c} {
        F(s - 1, i - 1, c - j) + t_intra((L_{i}, ..., L_{k}), j, DeviceMesh, s)
    }
    T(t_{max}) = min_{s, c}{F(s, K, c); t_{max}} + (G - 1) * t_{max}
    """
    assert num_stages > 1, "Only support tuning for num_stages > 1"
    num_layers = config.num_layers

    # Calculate the t_intra for base units for dp
    # features_np: (num_stages, stage_idx, num_layers, num_ckpt_layers)
    #                                      ^^^^^^^^^^  ^^^^^^^^^^^^^^^
    # samples for pre-layer and post-layer: (num_layers, num_layers + 1)
    pre_stage_latencies, pre_stage_solutions = calculate_stage_latencies(
        block_layer_info=block_layer_info,
        num_stages=num_stages,
        num_layers=num_layers,
        num_nodes_per_stage=num_nodes_per_stage,
        num_gpus_per_node_per_stage=num_gpus_per_node_per_stage,
        gradient_accumulation_steps=gradient_accumulation_steps,
        config=config,
        pre_layer_info=pre_layer_info,
        post_layer_info=None,
    )
    post_stage_latencies, post_stage_solutions = calculate_stage_latencies(
        block_layer_info=block_layer_info,
        num_stages=num_stages,
        num_layers=num_layers,
        num_nodes_per_stage=num_nodes_per_stage,
        num_gpus_per_node_per_stage=num_gpus_per_node_per_stage,
        gradient_accumulation_steps=gradient_accumulation_steps,
        config=config,
        pre_layer_info=None,
        post_layer_info=post_layer_info,
    )

    # * For block layers:
    # features_np: (num_stages, stage_idx, num_layers, num_ckpt_layers)
    #                           ^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^^^^
    # samples for block-layer: (num_stages - 1, num_layers, num_layers + 1)
    block_stage_latencies, block_stage_solutions = calculate_stage_latencies(
        block_layer_info=block_layer_info,
        num_stages=num_stages,
        num_layers=num_layers,
        num_nodes_per_stage=num_nodes_per_stage,
        num_gpus_per_node_per_stage=num_gpus_per_node_per_stage,
        gradient_accumulation_steps=gradient_accumulation_steps,
        config=config,
        pre_layer_info=None,
        post_layer_info=None,
    )

    # Fill the dp table
    def t_intra(stage_idx, start, end, ckpt_layers):
        """return the intra-stage latency for the layers in [start, end]
        all the layers consist of (pre_layer, *block_layers, post_layer)
        """
        cur_num_layers = end - start + 1
        if start == 1:
            return pre_stage_latencies[cur_num_layers - 1, ckpt_layers]
        elif end == num_layers:
            return post_stage_latencies[cur_num_layers - 1, ckpt_layers]
        else:
            return block_stage_latencies[stage_idx, cur_num_layers - 1, ckpt_layers]

    # Concatenate all latencies
    best_total_latency = float("inf")
    best_solution = None
    all_possible_stage_latencies = np.concatenate(
        [
            pre_stage_latencies.flatten(),
            post_stage_latencies.flatten(),
            block_stage_latencies.flatten(),
        ]
    )
    all_possible_stage_latencies = np.sort(np.unique(all_possible_stage_latencies))
    last_max_stage_latency = 0
    gap = 1
    for max_stage_latency in tqdm(all_possible_stage_latencies):
        if max_stage_latency * gradient_accumulation_steps >= best_total_latency:
            break
        if max_stage_latency - last_max_stage_latency < gap:
            continue
        latency = layer_partition_dp(
            t_intra=t_intra,
            t_max=max_stage_latency,
            num_stages=num_stages,
            num_layers=num_layers,
        )
        if latency < best_total_latency:
            best_total_latency = latency
        last_max_stage_latency = max_stage_latency

    return best_total_latency


_DISABLE_NUMBA = False


def numba_jit(func):
    from numba import jit  # pylint: disable=import-outside-toplevel

    jitted_func = jit(nopython=True)(func)

    def wrapper(*args, **kwargs):
        if _DISABLE_NUMBA:
            return func(*args, **kwargs)
        return jitted_func(*args, **kwargs)

    return wrapper


def layer_partition_dp(t_intra, t_max, num_stages, num_layers):
    dp_table = np.full(
        (num_stages, num_layers + 1, num_layers + 1), np.inf, dtype=float
    )
    dp_table[0, 0, 0] = 0
    for s in range(num_stages):
        for k in range(1, num_layers + 1):
            for c in range(k + 1):
                dp_table[s, k, c] = min(
                    [
                        dp_table[s - 1, i - 1, c - j] + t_intra(s - 1, i, k, j)
                        for i in range(1, k + 1)
                        for j in range(c + 1)
                        if t_intra(s - 1, i, k, j) <= t_max
                    ]
                    + [float("inf")]
                )

    best_total_latency = np.min(dp_table[:, -1, :])
    return best_total_latency


def build_and_tune_optimization_problem_with_pp_fixed(
    block_layer_info: LayerInfo,
    num_stages: int,
    block_layer_partition: List[int],
    num_nodes_per_stage: int,
    num_gpus_per_node_per_stage: int,
    gradient_accumulation_steps: int,
    config: MistConfig,
    pre_layer_info: Optional[LayerInfo] = None,
    post_layer_info: Optional[LayerInfo] = None,
):
    """Given the pipeline partition, build and tune the optimization problem.

    The parameters needed to be tuned are:
    - the number of ckpt layers for each stage
    - the strategies for each stages
    """
    assert num_stages > 1, "Only support tuning for num_stages > 1"
    stage_latencies = [None for _ in range(num_stages)]
    stage_features = [None for _ in range(num_stages)]

    # Calculate the t_intra for base units for dp
    # features_np: (num_stages, stage_idx, num_layers, num_ckpt_layers)
    #                                                  ^^^^^^^^^^^^^^^
    # samples for pre-layer and post-layer: (num_layer, num_layers + 1)
    # samples for block-layer: (num_layers, num_layers + 1, num_stages)

    for stage_idx in range(num_stages):
        is_first_stage = stage_idx == 0
        is_last_stage = stage_idx == num_stages - 1
        curr_num_layers = block_layer_partition[stage_idx]
        features_np = np.zeros((curr_num_layers + 1, 4), dtype=float)
        features_np[:, 0] = num_stages
        features_np[:, 1] = stage_idx
        features_np[:, 2] = curr_num_layers
        features_np[:, 3] = np.arange(curr_num_layers + 1)
        curr_stage_latencies, curr_stage_features = batched_tune_best_latency_for_stage(
            block_layer_info=block_layer_info,
            features_np=features_np,
            num_nodes=num_nodes_per_stage,
            num_gpus_per_node=num_gpus_per_node_per_stage,
            gradient_accumulation_steps=gradient_accumulation_steps,
            config=config,
            pre_layer_info=pre_layer_info if is_first_stage else None,
            post_layer_info=post_layer_info if is_last_stage else None,
        )
        best_curr_stage_latency = curr_stage_latencies.min()
        best_curr_stage_feature = curr_stage_features[curr_stage_latencies.argmin()]
        stage_latencies[stage_idx] = best_curr_stage_latency
        stage_features[stage_idx] = best_curr_stage_feature

    pipe_latency = sum(stage_latencies) + max(stage_latencies) * (
        gradient_accumulation_steps - 1
    )

    return pipe_latency, stage_features
