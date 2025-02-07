import itertools
import random
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


def get_solution_from_intra_stage_tuner_output(
    intra_stage_results: IntraStageTunerOutput,
    num_layers: int,
    gradient_accumulation_steps_index: int,
    pre_saved_micro_batches: int,
    layer_start: int,
    layer_end: int,
    device_mesh_idx: int,
    num_ckpt_layers: int,
):
    g = gradient_accumulation_steps_index
    q = device_mesh_idx
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


def inter_stage_tune(
    num_layers: int,
    num_devices: int,
    device_mesh_candidates: List[Tuple[int, int]],
    gradient_accumulation_steps_candidates: List[int],
    intra_stage_results: IntraStageTunerOutput,
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

    best_cost = float("inf")
    best_inter_solution = None
    best_grad_accumu_steps = None

    gradient_accumulation_steps_to_index = {
        g: i for i, g in enumerate(gradient_accumulation_steps_candidates)
    }
    device_mesh_to_index = {q: i for i, q in enumerate(device_mesh_candidates)}

    pbar_gard_accumu_steps = tqdm(
        gradient_accumulation_steps_candidates,
        desc="Inter-stage Tuning GradAccu",
        position=0,
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
            num_devices=num_devices,
            device_mesh_candidates=device_mesh_candidates,
            gradient_accumulation_steps=gradient_accumulation_steps,
            costs=cur_costs,
            costs_with_pre=cur_costs_with_pre,
            costs_with_post=cur_costs_with_post,
            costs_no_pp=cur_costs_no_pp,
            best_cost=best_cost,
        )
        if cur_best_cost < best_cost:
            best_cost = cur_best_cost
            best_grad_accumu_steps = gradient_accumulation_steps
            best_inter_solution = cur_best_inter_solution

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
            device_mesh_idx=device_mesh_to_index[(n, m)],
            num_ckpt_layers=ckpt_layers,
        )
        curr_intra_solution = curr_intra_solution.tolist()
        curr_intra_solution[:6] = [int(x) for x in curr_intra_solution[:6]]
        curr_intra_solution = tuple(curr_intra_solution)
        best_solution[1].append((curr_inter_solution, curr_intra_solution))

    return best_cost, best_solution


def inter_stage_tune_with_fixed_grad_accumu(
    num_layers: int,
    num_devices: int,
    device_mesh_candidates: List[Tuple[int, int]],
    gradient_accumulation_steps: int,
    costs: np.ndarray,
    costs_with_pre: np.ndarray,
    costs_with_post: np.ndarray,
    costs_no_pp: np.ndarray,
    best_cost=float("inf"),
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
            num_devices=num_devices,
            device_mesh_candidates=device_mesh_candidates,
            gradient_accumulation_steps=gradient_accumulation_steps,
            costs=costs,
            costs_with_pre=costs_with_pre,
            costs_with_post=costs_with_post,
            costs_no_pp=costs_no_pp,
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
    num_devices: int,
    device_mesh_candidates: List[Tuple[int, int]],
    gradient_accumulation_steps: int,
    costs: np.ndarray,
    costs_with_pre: np.ndarray,
    costs_with_post: np.ndarray,
    costs_no_pp: np.ndarray,
):
    """
    Suppose he layer sequence is l_1, l_2, ..., l_K.
    F(s, k, d; t_max) means slicing the layers `(l_k, ..., l_K)` into `s` stages,
    and putting them into `d` devices with `e` layers being checkpointed.

    F(s, k, d; t_max) = min_{k <= i <= K, 0 <= j <= e, n * m <= d}
    {
        t_intra(s, (l_k, ..., l_i), Mesh(n, m), j) +
        F(s - 1, i + 1, d - n * m; t_max)
        if t_intra(s, (l_k, ..., l_i), Mesh(n, m), j) <= t_max
    }
    Here s, k, d are 1-based indices; and e is 0-based index.
    F(0, K + 1, 0; t_max) = 0

    T(t_max) = min_{1 <= s <= MAX_POSSIBLE_S}
    {
        F(s, 0, D; t_max) + (G - 1) * t_max
    }
    """
    g = gradient_accumulation_steps

    # Note: layer is 0-based index
    def t_intra(s, l_start, l_end, q, c):
        """
        s: stage_idx from the last stage
        l_start: start_layer_idx
        l_end: end_layer_idx
        q: device_mesh_idx
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

    max_possible_num_stages = (
        min(max([n * m for n, m in device_mesh_candidates]), num_layers) // 2
    )
    f = np.full(
        (max_possible_num_stages + 1, num_layers + 1, num_devices + 1),
        np.inf,
        dtype=np.float32,
    )
    f_stage_max = np.full(
        (max_possible_num_stages + 1, num_layers + 1, num_devices + 1),
        0,
        dtype=np.float32,
    )
    f_argmin = np.full(
        (max_possible_num_stages + 1, num_layers + 1, num_devices + 1, 4),
        -1,
        dtype=np.int32,
    )
    f[0, num_layers, 0] = 0

    # ==============================
    # dp loop
    # ==============================
    for s in range(1, max_possible_num_stages + 1):
        for k in range(num_layers - s, -1, -1):
            for d in range(1, num_devices + 1):
                precomputed_device_mesh_index_candidates = [
                    (q, (n, m))
                    for q, (n, m) in enumerate(device_mesh_candidates)
                    if n * m <= d
                ]
                # ==============================
                # inner loop
                # ==============================
                for q, (n, m) in precomputed_device_mesh_index_candidates:
                    # (l_k, ..., l_i), and (l_{i + 1}, ..., l_K)
                    for i in range(k, num_layers):
                        for j in range(num_layers - i + 1):
                            stage_cost = t_intra(s, k, i, q, j)
                            new_cost = stage_cost + f[s - 1, i + 1, d - n * m]
                            if (
                                stage_cost <= max_stage_latency
                                and new_cost < f[s, k, d]
                            ):
                                f[s, k, d] = new_cost
                                f_stage_max[s, k, d] = max(
                                    f_stage_max[s - 1, i + 1, d - n * m], stage_cost
                                )
                                f_argmin[s, k, d] = (i, n, m, j)

    best_s = -1
    best_total_cost = np.inf
    for s in range(1, max_possible_num_stages + 1):
        if f[s, 0, num_devices] < best_total_cost:
            best_s = s
            best_total_cost = f[s, 0, num_devices]

    if np.isinf(best_total_cost):
        return np.inf, None

    total_cost = best_total_cost + (g - 1) * f_stage_max[best_s, 0, num_devices]
    current_s = best_s
    current_layer = 0
    current_devices = num_devices

    res = []
    while current_s > 0 and current_layer < num_layers and current_devices > 0:
        current_end_layer, n, m, ckpt_layers = f_argmin[
            current_s, current_layer, current_devices
        ]
        assert current_end_layer != -1 and current_devices != -1
        res.append(((current_layer, current_end_layer), (n, m), ckpt_layers))
        current_s -= 1
        current_layer = current_end_layer + 1
        current_devices -= n * m
    assert current_s == 0 and current_layer == num_layers and current_devices == 0

    return total_cost, res
