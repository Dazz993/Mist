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
        return intra_stage_results.solutions_no_pp[g, q, c]
    elif layer_start == 0:
        return intra_stage_results.solutions_with_pre[g, q, p, l - 1, c]
    elif layer_end == num_layers - 1:
        return intra_stage_results.solutions_with_post[g, q, l - 1, c]
    else:
        return intra_stage_results.solutions[g, q, p, l - 1, c]


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
    device_mesh_candidates: List[Tuple[int, int]],
    gradient_accumulation_steps_candidates: List[int],
    intra_stage_results: IntraStageTunerOutput,
    config: MistConfig,
):
    if not isinstance(intra_stage_results, IntraStageTunerOutput):
        raise TypeError(
            f"intra_stage_results must be an instance of IntraStageTunerOutput, got {type(intra_stage_results)}"
        )

    if intra_stage_results.costs is None:
        raise ValueError("intra_stage_results.costs must not be None")

    if intra_stage_results.costs_with_pre is None:
        raise ValueError("intra_stage_results.costs_with_pre must not be None")

    if intra_stage_results.costs_with_post is None:
        raise ValueError("intra_stage_results.costs_with_post must not be None")

    costs = intra_stage_results.costs
    costs_with_pre = intra_stage_results.costs_with_pre
    costs_with_post = intra_stage_results.costs_with_post
    costs_no_pp = intra_stage_results.costs_no_pp

    costs = costs.astype(np.float32)
    costs_with_pre = costs_with_pre.astype(np.float32)
    costs_with_post = costs_with_post.astype(np.float32)
    costs_no_pp = costs_no_pp.astype(np.float32)

    gradient_accumulation_steps_to_index = {
        g: i
        for i, g in enumerate(
            intra_stage_results.gradient_accumulation_steps_candidates
        )
    }
    device_mesh_to_index = {
        q: i for i, q in enumerate(intra_stage_results.device_mesh_candidates)
    }

    # Convert candidates to index
    device_mesh_index_candidates = [
        device_mesh_to_index[q] for q in device_mesh_candidates
    ]

    best_cost = float("inf")
    best_g = None
    best_g_index = None
    best_num_stages = None
    best_num_ckpt_layers = None

    best_inter_solution = None

    pbar_gard_accumu_steps = tqdm(
        gradient_accumulation_steps_candidates,
        file=sys.stdout,
        dynamic_ncols=False,
        desc="Inter-stage Tuning GradAccu",
        position=0,
        disable=DISABLE_TQDM,
    )
    for gradient_accumulation_steps in pbar_gard_accumu_steps:
        pbar_gard_accumu_steps.set_postfix(
            {"gradient_accumulation_steps": gradient_accumulation_steps}
        )
        g_index = gradient_accumulation_steps_to_index[gradient_accumulation_steps]
        cur_costs = costs[g_index]
        cur_costs_with_pre = costs_with_pre[g_index]
        cur_costs_with_post = costs_with_post[g_index]
        cur_costs_no_pp = costs_no_pp[g_index]

        (
            cur_best_cost,
            cur_best_inter_solution,
        ) = inter_stage_tune_with_fixed_grad_accumu(
            num_layers=num_layers,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            device_mesh_candidates=device_mesh_candidates,
            device_mesh_index_candidates=device_mesh_index_candidates,
            gradient_accumulation_steps=gradient_accumulation_steps,
            costs=cur_costs,
            costs_with_pre=cur_costs_with_pre,
            costs_with_post=cur_costs_with_post,
            costs_no_pp=cur_costs_no_pp,
            best_cost=best_cost,
            activation_checkpointing_tuning_enabled=config.tuning.activation_checkpointing_tuning_enabled,
        )
        if cur_best_cost < best_cost:
            best_cost = cur_best_cost
            best_grad_accumu_steps = gradient_accumulation_steps
            best_inter_solution = cur_best_inter_solution

    if best_cost == float("inf"):
        return float("inf"), None

    # Map the best solution to get the best solution of both inter-stage and intra-stage
    best_solution = [best_grad_accumu_steps, []]
    num_stages = len(best_inter_solution)
    for i, curr_inter_solution in enumerate(best_inter_solution):
        ((start_layer, end_layer), (n, m), ckpt_layers) = curr_inter_solution
        warmup = num_stages - i - 1
        curr_intra_solution = get_solution_from_intra_stage_tuner_output(
            intra_stage_results=intra_stage_results,
            num_layers=num_layers,
            gradient_accumulation_steps_index=gradient_accumulation_steps_to_index[
                best_grad_accumu_steps
            ],
            pre_saved_micro_batches=warmup,
            layer_start=start_layer,
            layer_end=end_layer,
            device_mesh_index=device_mesh_to_index[(n, m)],
            num_ckpt_layers=ckpt_layers,
        )
        curr_intra_solution = curr_intra_solution.tolist()
        curr_intra_solution[:6] = [int(x) for x in curr_intra_solution[:6]]
        curr_intra_solution = tuple(curr_intra_solution)
        best_solution[1].append((curr_inter_solution, curr_intra_solution))

    return best_cost, best_solution


def inter_stage_tune_with_fixed_grad_accumu(
    num_layers: int,
    num_nodes: int,
    num_gpus_per_node: int,
    device_mesh_candidates: List[Tuple[int, int]],
    device_mesh_index_candidates: List[int],
    gradient_accumulation_steps: int,
    costs: np.ndarray,
    costs_with_pre: np.ndarray,
    costs_with_post: np.ndarray,
    costs_no_pp: np.ndarray,
    best_cost=float("inf"),
    activation_checkpointing_tuning_enabled: bool = True,
):
    best_solution = None
    last_max_stage_cost = 0
    gap = 1e-6
    # gap = 5e-4

    # Get and sort all possible stage latencies
    all_possible_stage_latencies = np.concatenate(
        (costs.flatten(), costs_with_pre.flatten(), costs_with_post.flatten())
    )
    all_possible_stage_latencies = np.sort(np.unique(all_possible_stage_latencies))
    all_possible_stage_latencies = all_possible_stage_latencies[
        all_possible_stage_latencies != np.inf
    ]

    pbar_max_stage_cost = tqdm(
        all_possible_stage_latencies,
        desc="Inter-stage Tuning MaxStageCost",
        position=1,
        leave=False,
        disable=DISABLE_TQDM,
    )
    for max_stage_latency in pbar_max_stage_cost:
        pbar_max_stage_cost.set_postfix({"max_stage_latency": max_stage_latency})
        if max_stage_latency * gradient_accumulation_steps >= best_cost:
            break
        if max_stage_latency - last_max_stage_cost < gap:
            continue

        cost, solution = inter_stage_tune_dp_impl(
            max_stage_latency=max_stage_latency,
            num_layers=num_layers,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            device_mesh_candidates=device_mesh_candidates,
            device_mesh_index_candidates=device_mesh_index_candidates,
            gradient_accumulation_steps=gradient_accumulation_steps,
            costs=costs,
            costs_with_pre=costs_with_pre,
            costs_with_post=costs_with_post,
            costs_no_pp=costs_no_pp,
            activation_checkpointing_tuning_enabled=activation_checkpointing_tuning_enabled,
        )
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
        last_max_stage_cost = max_stage_latency

    return best_cost, best_solution


@njit(fastmath=True, cache=True, parallel=False)
def inter_stage_tune_dp_impl(
    max_stage_latency: float,
    num_layers: int,
    num_nodes: int,
    num_gpus_per_node: int,
    device_mesh_candidates: List[Tuple[int, int]],
    device_mesh_index_candidates: List[int],
    gradient_accumulation_steps: int,
    costs: np.ndarray,
    costs_with_pre: np.ndarray,
    costs_with_post: np.ndarray,
    costs_no_pp: np.ndarray,
    activation_checkpointing_tuning_enabled: bool = True,
):
    """
    Suppose the layer sequence is l_1, l_2, ..., l_K. And the number of stages is S.
    F(s, k; t_max) means slicing the layers `(l_k, ..., l_K)` into `s` stages,
    and putting them into a fixed device mesh with `e` layers being checkpointed.

    F(s, k; t_max) = min_{k <= i <= K, 0 <= j <= i - k + 1}
    {
        t_intra(s, (l_k, ..., l_i), Mesh, j) +
        F(s - 1, i + 1; t_max)
        if t_intra(s, (l_k, ..., l_i), Mesh, j) <= t_max
    }
    Here s, k are 1-based indices; and e is 0-based index.
    F(0, K + 1; t_max) = 0

    T(t_max) = min_{1 <= s <= MAX_POSSIBLE_S}
    {
        F(s, 0; t_max) + (G - 1) * t_max
    }
    """
    g = gradient_accumulation_steps

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

        # Find the device mesh index
        device_mesh_index = -1
        for q, (n, m) in enumerate(device_mesh_candidates):
            if n == num_nodes and m == num_gpus_per_node:
                device_mesh_index = q
        assert device_mesh_index != -1

        return (num_nodes, num_gpus_per_node), device_mesh_index

    # Note: layer is 0-based index
    def t_intra(s, l_start, l_end, q, c):
        """
        s: stage_idx from the last stage
        l_start: start_layer_idx
        l_end: end_layer_idx
        q: device_mesh_index
        n: num_nodes
        m: num_gpus_per_node
        c: curr_num_ckpt_layers
        l: curr_num_layers
        p: pre_saved_micro_batches
        """
        l = l_end - l_start + 1
        p = s - 1

        if l_start == 0 and l_end == num_layers - 1:
            return costs_no_pp[q, c]
        elif l_start == 0:
            return costs_with_pre[q, p, l - 1, c]
        elif l_end == num_layers - 1:
            return costs_with_post[q, l - 1, c]
        else:
            return costs[q, p, l - 1, c]

    global_best_total_cost = np.inf
    global_best_res = None
    max_possible_num_stages = num_layers

    # Outer loop: num_stages
    for S in range(1, max_possible_num_stages + 1):
        device_mesh_info = _uniform_device_mesh_partition(
            S, num_nodes, num_gpus_per_node
        )
        if device_mesh_info is None:
            continue
        (n, m), q = device_mesh_info
        # ==============================
        # dp loop
        # ==============================
        f = np.full(
            (S + 1, num_layers + 1),
            np.inf,
            dtype=np.float32,
        )
        f_stage_max = np.full(
            (S + 1, num_layers + 1),
            0,
            dtype=np.float32,
        )
        f_argmin = np.full(
            (S + 1, num_layers + 1, 2),
            -1,
            dtype=np.int32,
        )
        f[0, num_layers] = 0

        for s in range(1, S + 1):
            for k in range(num_layers - s, -1, -1):
                # ==============================
                # inner loop
                # ==============================
                # (l_k, ..., l_i), and (l_{i + 1}, ..., l_K)
                for i in range(k, num_layers):
                    if activation_checkpointing_tuning_enabled:
                        ckpt_candidates = list(range(i - k + 1 + 1))
                    else:
                        ckpt_candidates = [i - k + 1]
                    for j in ckpt_candidates:
                        stage_cost = t_intra(s, k, i, q, j)
                        new_cost = stage_cost + f[s - 1, i + 1]
                        if stage_cost <= max_stage_latency and new_cost < f[s, k]:
                            f[s, k] = new_cost
                            f_stage_max[s, k] = max(
                                f_stage_max[s - 1, i + 1], stage_cost
                            )
                            f_argmin[s, k] = (i, j)

        best_s = -1
        best_total_cost = np.inf
        for s in range(1, S + 1):
            if f[s, 0] < best_total_cost:
                best_s = s
                best_total_cost = f[s, 0]

        if np.isinf(best_total_cost):
            continue

        total_cost = best_total_cost + (g - 1) * f_stage_max[best_s, 0]
        current_s = best_s
        current_layer = 0

        res = []
        while current_s > 0 and current_layer < num_layers:
            current_end_layer, ckpt_layers = f_argmin[current_s, current_layer]
            assert current_end_layer != -1
            res.append(((current_layer, current_end_layer), (n, m), ckpt_layers))
            current_s -= 1
            current_layer = current_end_layer + 1
        assert current_s == 0 and current_layer == num_layers

        if total_cost < global_best_total_cost:
            global_best_total_cost = total_cost
            global_best_res = res

    return global_best_total_cost, global_best_res
