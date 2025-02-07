import itertools
import os
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, cache
from time import perf_counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional, TYPE_CHECKING, Any

from numba import jit, njit, prange

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
from mist.analyzer.intra_tuning import IntraStageTunerOutput
from mist.analyzer.model_analyzer import ModelAnalyzer
from mist.logger import get_logger
from mist.utils.pipeline_parallel import calculate_num_warmup_and_1f1b_phases
from mist.tuning.batched_model_optim_prob import ModelGranularityOptimProb
from mist.analyzer.batched_module_analyzer import batched_tune_best_latency_for_stage

logger = get_logger(__name__)

POWER_OF_TWO = [2**i for i in range(1, 15)]

_DISABLE_NUMBA = False

DISABLE_TQDM = False
if str(os.getenv("DISABLE_TQDM", False)).lower() == "true":
    DISABLE_TQDM = True


def get_solution_from_intra_stage_tuner_output(
    intra_stage_results: IntraStageTunerOutput,
    num_layers: int,
    gradient_accumulation_steps_index: int,
    pre_saved_micro_batches: int,
    layer_start: int,
    layer_end: int,
    device_mesh_index: int,
    num_ckpt_layers: int,
):
    g = gradient_accumulation_steps_index
    q = device_mesh_index
    p = pre_saved_micro_batches
    l = layer_end - layer_start + 1
    c = num_ckpt_layers

    if layer_start == 0 and layer_end == num_layers - 1:
        return intra_stage_results.solutions_no_pp[g, q, c, 0]
    elif layer_start == 0:
        return intra_stage_results.solutions_with_pre[g, q, p, l - 1, c, 0]
    elif layer_end == num_layers - 1:
        return intra_stage_results.solutions_with_post[g, q, l - 1, c, 0]
    else:
        return intra_stage_results.solutions[g, q, p, l - 1, c, 0]


def get_cost_from_intra_stage_tuner_output(
    intra_stage_results: IntraStageTunerOutput,
    num_layers: int,
    gradient_accumulation_steps_index: int,
    pre_saved_micro_batches: int,
    layer_start: int,
    layer_end: int,
    device_mesh_index: int,
    num_ckpt_layers: int,
):
    g = gradient_accumulation_steps_index
    q = device_mesh_index
    p = pre_saved_micro_batches
    l = layer_end - layer_start + 1
    c = num_ckpt_layers

    if layer_start == 0 and layer_end == num_layers - 1:
        return intra_stage_results.costs_no_pp[g, q, c]
    elif layer_start == 0:
        return intra_stage_results.costs_with_pre[g, q, p, l - 1, c]
    elif layer_end == num_layers - 1:
        return intra_stage_results.costs_with_post[g, q, l - 1, c]
    else:
        return intra_stage_results.costs[g, q, p, l - 1, c]


def _uniform_layer_partition(num_layers: int, num_stages: int) -> List[int]:
    """Uniformly partition the layers into stages."""
    num_layers_per_stage = num_layers // num_stages
    layer_partition = [num_layers_per_stage] * num_stages
    # Add layers to the last few stages
    for i in range(num_layers % num_stages):
        layer_partition[-i - 1] += 1
    return layer_partition


def _uniform_device_mesh_partition(num_stages, num_nodes, num_gpus_per_node):
    if num_nodes > num_stages:
        if num_nodes % num_stages != 0:
            return None
        num_nodes = num_nodes // num_stages
        num_gpus_per_node = num_gpus_per_node
    else:
        if (
            num_stages % num_nodes != 0
            or num_gpus_per_node % (num_stages // num_nodes) != 0
        ):
            return None
        num_gpus_per_node = num_gpus_per_node // (num_stages // num_nodes)
        num_nodes = 1

    return num_nodes, num_gpus_per_node


def inter_stage_tune(
    num_layers: int,
    num_nodes: int,
    num_gpus_per_node: int,
    gradient_accumulation_steps_candidates: List[int],
    intra_stage_results: IntraStageTunerOutput,
    config: MistConfig,
):
    if not isinstance(intra_stage_results, IntraStageTunerOutput):
        raise TypeError(
            f"intra_stage_results must be an instance of IntraStageTunerOutput, got {type(intra_stage_results)}"
        )

    activation_checkpointing_tuning_enabled = (
        config.tuning.activation_checkpointing_tuning_enabled
    )
    simple_activation_checkpointing_tuning_enabled = (
        config.tuning.tuning_granularity == "uniform-pp-simple-heuristic-mem-opt"
    )

    costs_stable = intra_stage_results.costs_stable
    costs_stable_with_pre = intra_stage_results.costs_stable_with_pre
    costs_stable_with_post = intra_stage_results.costs_stable_with_post
    costs_stable_no_pp = intra_stage_results.costs_stable_no_pp
    costs_stable = costs_stable.astype(np.float32)
    costs_stable_with_pre = costs_stable_with_pre.astype(np.float32)
    costs_stable_with_post = costs_stable_with_post.astype(np.float32)
    costs_stable_no_pp = costs_stable_no_pp.astype(np.float32)

    costs_delta = intra_stage_results.costs_delta
    costs_delta_with_pre = intra_stage_results.costs_delta_with_pre
    costs_delta_with_post = intra_stage_results.costs_delta_with_post
    costs_delta_no_pp = intra_stage_results.costs_delta_no_pp
    costs_delta = costs_delta.astype(np.float32)
    costs_delta_with_pre = costs_delta_with_pre.astype(np.float32)
    costs_delta_with_post = costs_delta_with_post.astype(np.float32)
    costs_delta_no_pp = costs_delta_no_pp.astype(np.float32)

    gradient_accumulation_steps_to_index = {
        g: i
        for i, g in enumerate(
            intra_stage_results.gradient_accumulation_steps_candidates
        )
    }
    device_mesh_to_index = {
        q: i for i, q in enumerate(intra_stage_results.device_mesh_candidates)
    }

    best_cost = float("inf")
    best_g = None
    best_g_index = None
    best_num_stages = None
    best_num_ckpt_layers = None
    num_stages_candidates = list(range(1, num_layers + 1))
    for num_stages in num_stages_candidates:
        if num_nodes * num_gpus_per_node % num_stages != 0:
            continue
        layer_partition = _uniform_layer_partition(num_layers, num_stages)
        n, m = _uniform_device_mesh_partition(num_stages, num_nodes, num_gpus_per_node)
        device_mesh_index = device_mesh_to_index[(n, m)]
        gradient_accumulation_steps_index_candidates = [
            gradient_accumulation_steps_to_index[g]
            for g in gradient_accumulation_steps_candidates
        ]
        (
            curr_best_cost,
            curr_best_g,
            curr_best_g_index,
            curr_best_num_ckpt_layers,
        ) = inter_stage_tune_with_fixed_inter_partition(
            num_layers=num_layers,
            num_stages=num_stages,
            layer_partition=layer_partition,
            num_nodes_per_stage=n,
            num_gpus_per_node_per_stage=m,
            device_mesh_index=device_mesh_index,
            gradient_accumulation_steps_candidates=gradient_accumulation_steps_candidates,
            gradient_accumulation_steps_index_candidates=gradient_accumulation_steps_index_candidates,
            costs_stable=costs_stable,
            costs_stable_with_pre=costs_stable_with_pre,
            costs_stable_with_post=costs_stable_with_post,
            costs_stable_no_pp=costs_stable_no_pp,
            costs_delta=costs_delta,
            costs_delta_with_pre=costs_delta_with_pre,
            costs_delta_with_post=costs_delta_with_post,
            costs_delta_no_pp=costs_delta_no_pp,
            activation_checkpointing_tuning_enabled=activation_checkpointing_tuning_enabled,
            simple_activation_checkpointing_tuning_enabled=simple_activation_checkpointing_tuning_enabled,
        )

        # print(f"num_stages: {num_stages}, best_cost: {curr_best_cost}")
        if curr_best_cost < best_cost:
            best_cost = curr_best_cost
            best_g = curr_best_g
            best_g_index = curr_best_g_index
            best_num_stages = num_stages
            best_num_ckpt_layers = curr_best_num_ckpt_layers

    # Conclude the best solution
    solutions = [best_g, []]
    layer_partition = _uniform_layer_partition(num_layers, best_num_stages)
    n, m = _uniform_device_mesh_partition(best_num_stages, num_nodes, num_gpus_per_node)
    device_mesh_index = device_mesh_to_index[(n, m)]
    layer_starts = np.cumsum(layer_partition) - layer_partition
    layer_ends = np.cumsum(layer_partition) - 1
    for stage_index in range(best_num_stages):
        curr_num_layers = layer_partition[stage_index]
        layer_start = layer_starts[stage_index]
        layer_end = layer_ends[stage_index]
        curr_num_ckpt_layers = best_num_ckpt_layers[stage_index]
        curr_intra_solution = get_solution_from_intra_stage_tuner_output(
            intra_stage_results=intra_stage_results,
            num_layers=num_layers,
            gradient_accumulation_steps_index=best_g_index,
            pre_saved_micro_batches=best_num_stages - stage_index - 1,
            layer_start=layer_start,
            layer_end=layer_end,
            device_mesh_index=device_mesh_index,
            num_ckpt_layers=curr_num_ckpt_layers,
        )
        curr_intra_solution = curr_intra_solution.tolist()
        curr_intra_solution[:6] = [int(x) for x in curr_intra_solution[:6]]
        curr_intra_solution = tuple(curr_intra_solution)
        curr_inter_solution = (
            (layer_start, layer_end),
            (n, m),
            curr_num_ckpt_layers,
            0,
        )
        solutions[1].append((curr_inter_solution, curr_intra_solution))
        # print(
        #     f"num_stages: {best_num_stages}, stage_index: {stage_index}, layer_start: {layer_start}, layer_end: {layer_end}"
        # )

    return best_cost, solutions


# @njit(fastmath=True, cache=True, parallel=False)
def inter_stage_tune_with_fixed_inter_partition(
    num_layers: int,
    num_stages: int,
    layer_partition: List[int],
    num_nodes_per_stage: int,
    num_gpus_per_node_per_stage: int,
    device_mesh_index: int,
    gradient_accumulation_steps_candidates: List[int],
    gradient_accumulation_steps_index_candidates: List[int],
    costs_stable: np.ndarray,
    costs_stable_with_pre: np.ndarray,
    costs_stable_with_post: np.ndarray,
    costs_stable_no_pp: np.ndarray,
    costs_delta: np.ndarray,
    costs_delta_with_pre: np.ndarray,
    costs_delta_with_post: np.ndarray,
    costs_delta_no_pp: np.ndarray,
    activation_checkpointing_tuning_enabled: bool,
    simple_activation_checkpointing_tuning_enabled: bool = False,
):
    n = num_nodes_per_stage
    m = num_gpus_per_node_per_stage
    # Here layer starts and layer ends are inclusive
    layer_partition = np.array(layer_partition)
    layer_starts = np.cumsum(layer_partition) - layer_partition
    layer_ends = np.cumsum(layer_partition) - 1

    def t_intra(
        gradient_accumulation_steps_index: int,
        stage_index: int,
        l_start: int,
        l_end: int,
        num_ckpt_layers: int,
    ):
        # l_start and l_end are inclusive
        curr_num_layers = l_end - l_start + 1
        g = gradient_accumulation_steps_index
        q = device_mesh_index
        p = num_stages - stage_index - 1
        l = curr_num_layers
        c = num_ckpt_layers

        if l_start == 0 and l_end == num_layers - 1:
            stable = costs_stable_no_pp[g, q, c, 0]
            delta = costs_delta_no_pp[g, q, c, 0]
        elif l_start == 0:
            stable = costs_stable_with_pre[g, q, p, l - 1, c, 0]
            delta = costs_delta_with_pre[g, q, p, l - 1, c, 0]
        elif l_end == num_layers - 1:
            stable = costs_stable_with_post[g, q, l - 1, c, 0]
            delta = costs_delta_with_post[g, q, l - 1, c, 0]
        else:
            stable = costs_stable[g, q, p, l - 1, c, 0]
            delta = costs_delta[g, q, p, l - 1, c, 0]

        return stable + delta / gradient_accumulation_steps_candidates[g]

    best_cost = np.inf
    best_g = None
    best_g_index = None
    best_num_ckpt_layers = None

    for g, g_index in zip(
        gradient_accumulation_steps_candidates,
        gradient_accumulation_steps_index_candidates,
    ):
        if simple_activation_checkpointing_tuning_enabled:
            for num_ckpt_layers in range(0, min(layer_partition) + 1):
                stage_latencies = []
                for stage_index in range(num_stages):
                    curr_layer_start = layer_starts[stage_index]
                    curr_layer_end = layer_ends[stage_index]
                    stage_latency = t_intra(
                        g_index,
                        stage_index,
                        curr_layer_start,
                        curr_layer_end,
                        num_ckpt_layers,
                    )
                    stage_latencies.append(stage_latency)

                stage_latency = max(stage_latencies) * (g - 1) + sum(stage_latencies)

                if np.inf in stage_latencies:
                    continue

                stage_latency = max(stage_latencies) * (g - 1) + sum(stage_latencies)
                if stage_latency < best_cost:
                    best_cost = stage_latency
                    best_g = g
                    best_g_index = g_index
                    best_num_ckpt_layers = [num_ckpt_layers] * num_stages

        else:
            stage_latencies = []
            curr_num_ckpt_layers = []
            for stage_index in range(num_stages):
                curr_layer_start = layer_starts[stage_index]
                curr_layer_end = layer_ends[stage_index]
                if not activation_checkpointing_tuning_enabled:
                    stage_best_num_ckpt_layers = curr_layer_end - curr_layer_start + 1
                    stage_best_intra_time = t_intra(
                        g_index,
                        stage_index,
                        curr_layer_start,
                        curr_layer_end,
                        stage_best_num_ckpt_layers,
                    )
                else:
                    stage_best_num_ckpt_layers = None
                    stage_best_intra_time = np.inf
                    for num_ckpt_layers in range(curr_layer_end - curr_layer_start + 2):
                        intra_time = t_intra(
                            g_index,
                            stage_index,
                            curr_layer_start,
                            curr_layer_end,
                            num_ckpt_layers,
                        )
                        if intra_time < stage_best_intra_time:
                            stage_best_intra_time = intra_time
                            stage_best_num_ckpt_layers = num_ckpt_layers

                stage_latencies.append(stage_best_intra_time)
                curr_num_ckpt_layers.append(stage_best_num_ckpt_layers)

            if np.inf in stage_latencies:
                continue

            stage_latency = max(stage_latencies) * (g - 1) + sum(stage_latencies)
            if stage_latency < best_cost:
                best_cost = stage_latency
                best_g = g
                best_g_index = g_index
                best_num_ckpt_layers = curr_num_ckpt_layers

    return best_cost, best_g, best_g_index, best_num_ckpt_layers
