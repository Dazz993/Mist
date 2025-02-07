import itertools
import multiprocessing
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
import pulp as pl
from pprint import pformat

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
from mist.tuning.batched_model_optim_prob import ModelGranularityOptimProb
from mist.analyzer.batched_module_analyzer import batched_tune_best_latency_for_stage
from mist.utils.pipeline_parallel import calculate_num_warmup_and_1f1b_phases
from mist.utils.pareto import sample_pareto_frontier, fill_redundant_samples

DISABLE_TQDM = False
if str(os.getenv("DISABLE_TQDM", False)).lower() == "true":
    DISABLE_TQDM = True

MAX_VALUE = 1000000.0

logger = get_logger()


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


def get_solution_from_intra_stage_tuner_output(
    intra_stage_results: IntraStageTunerOutput,
    num_layers: int,
    gradient_accumulation_steps_index: int,
    pre_saved_micro_batches: int,
    layer_start: int,
    layer_end: int,
    device_mesh_index: int,
    num_ckpt_layers: int,
    policy_index: int,
):
    g = gradient_accumulation_steps_index
    q = device_mesh_index
    p = pre_saved_micro_batches
    l = layer_end - layer_start + 1
    c = num_ckpt_layers
    k = policy_index

    if layer_start == 0 and layer_end == num_layers - 1:
        return intra_stage_results.solutions_no_pp[g, q, c, k]
    elif layer_start == 0:
        return intra_stage_results.solutions_with_pre[g, q, p, l - 1, c, k]
    elif layer_end == num_layers - 1:
        return intra_stage_results.solutions_with_post[g, q, l - 1, c, k]
    else:
        return intra_stage_results.solutions[g, q, p, l - 1, c, k]


def inter_stage_tune(
    num_layers: int,
    num_nodes: int,
    num_gpus_per_node: int,
    device_mesh_candidates: List[Tuple[int, int]],
    gradient_accumulation_steps_candidates: List[int],
    intra_stage_results: IntraStageTunerOutput,
    config: MistConfig,
) -> Tuple[float, Tuple[Any]]:
    if not isinstance(intra_stage_results, IntraStageTunerOutput):
        raise TypeError(
            f"intra_stage_results must be an instance of IntraStageTunerOutput, got {type(intra_stage_results)}"
        )

    if intra_stage_results.costs_stable is None:
        raise ValueError("intra_stage_results.costs must not be None")

    if intra_stage_results.costs_stable_with_pre is None:
        raise ValueError("intra_stage_results.costs_with_pre must not be None")

    if intra_stage_results.costs_stable_with_post is None:
        raise ValueError("intra_stage_results.costs_with_post must not be None")

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
    num_stages_candidates = list(range(1, num_layers + 1))

    best_cost = MAX_VALUE
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
    pbar_stages = tqdm(
        num_stages_candidates,
        file=sys.stdout,
        desc="Inter-stage Tuning Num Stages",
        position=1,
        leave=False,
        disable=DISABLE_TQDM,
    )
    for gradient_accumulation_steps in pbar_gard_accumu_steps:
        pbar_gard_accumu_steps.set_postfix(
            {"gradient_accumulation_steps": gradient_accumulation_steps}
        )
        g_index = gradient_accumulation_steps_to_index[gradient_accumulation_steps]
        for num_stages in pbar_stages:
            pbar_stages.set_postfix({"num_stages": num_stages})
            if num_nodes * num_gpus_per_node % num_stages != 0:
                continue
            n, m = _uniform_device_mesh_partition(
                num_stages, num_nodes, num_gpus_per_node
            )
            q_index = device_mesh_to_index[(n, m)]

            curr_costs_stable = costs_stable[g_index, q_index]
            curr_costs_stable_with_pre = costs_stable_with_pre[g_index, q_index]
            curr_costs_stable_with_post = costs_stable_with_post[g_index, q_index]
            curr_costs_stable_no_pp = costs_stable_no_pp[g_index, q_index]
            curr_costs_delta = costs_delta[g_index, q_index]
            curr_costs_delta_with_pre = costs_delta_with_pre[g_index, q_index]
            curr_costs_delta_with_post = costs_delta_with_post[g_index, q_index]
            curr_costs_delta_no_pp = costs_delta_no_pp[g_index, q_index]

            (
                cur_best_cost,
                cur_best_inter_solution,
            ) = inter_stage_tune_with_fixed_grad_accumu_and_device_partition(
                num_layers=num_layers,
                num_stages=num_stages,
                gradient_accumulation_steps=gradient_accumulation_steps,
                costs_stable=curr_costs_stable,
                costs_stable_with_pre=curr_costs_stable_with_pre,
                costs_stable_with_post=curr_costs_stable_with_post,
                costs_stable_no_pp=curr_costs_stable_no_pp,
                costs_delta=curr_costs_delta,
                costs_delta_with_pre=curr_costs_delta_with_pre,
                costs_delta_with_post=curr_costs_delta_with_post,
                costs_delta_no_pp=curr_costs_delta_no_pp,
                device_mesh=(n, m),
                best_cost=best_cost,
                activation_checkpointing_tuning_enabled=config.tuning.activation_checkpointing_tuning_enabled,
                layers_offset=getattr(config.tuning, "layers_offset", 2),
                fixed_ckpt_value=getattr(config, "fixed_ckpt_value", None),
            )
            if cur_best_cost < best_cost:
                best_cost = cur_best_cost
                best_g = gradient_accumulation_steps
                best_g_index = g_index
                best_num_stages = num_stages
                best_inter_solution = cur_best_inter_solution

    if best_cost == float("inf"):
        return float("inf"), None

    # ==================================================================================================
    best_solution = None
    # ==================================================================================================

    # Map the best solution to get the best solution of both inter-stage and intra-stage
    best_solution = [best_g, []]
    num_stages = len(best_inter_solution)
    assert num_stages == best_num_stages, f"{num_stages=} {best_num_stages=}"
    for i, curr_inter_solution in enumerate(best_inter_solution):
        ((start_layer, end_layer), (n, m), ckpt_layers, policy_index) = (
            curr_inter_solution
        )
        warmup = num_stages - i - 1
        curr_intra_solution = get_solution_from_intra_stage_tuner_output(
            intra_stage_results=intra_stage_results,
            num_layers=num_layers,
            gradient_accumulation_steps_index=best_g_index,
            pre_saved_micro_batches=warmup,
            layer_start=start_layer,
            layer_end=end_layer,
            device_mesh_index=device_mesh_to_index[(n, m)],
            num_ckpt_layers=ckpt_layers,
            policy_index=policy_index,
        )
        curr_intra_solution = curr_intra_solution.tolist()
        curr_intra_solution[:6] = [int(x) for x in curr_intra_solution[:6]]
        curr_intra_solution = tuple(curr_intra_solution)
        best_solution[1].append((curr_inter_solution, curr_intra_solution))

    return best_cost, best_solution


def inter_stage_tune_with_fixed_grad_accumu_and_device_partition(
    num_layers: int,
    num_stages: int,
    gradient_accumulation_steps: int,
    costs_stable: np.ndarray,
    costs_stable_with_pre: np.ndarray,
    costs_stable_with_post: np.ndarray,
    costs_stable_no_pp: np.ndarray,
    costs_delta: np.ndarray,
    costs_delta_with_pre: np.ndarray,
    costs_delta_with_post: np.ndarray,
    costs_delta_no_pp: np.ndarray,
    device_mesh: Tuple[int, int],
    best_cost=float("inf"),
    activation_checkpointing_tuning_enabled: bool = True,
    layers_offset: int = 2,
    fixed_ckpt_value: Optional[int] = None,
) -> Tuple[float, Tuple[Any]]:
    logger.info(
        f"Begin inter-stage tuning with S={num_stages}, G={gradient_accumulation_steps}, {device_mesh=}"
    )

    L_OFFSET = layers_offset
    L = num_layers
    S = num_stages
    G = gradient_accumulation_steps

    # Limit the number of layers to be tuned
    _averaged_num_layers = round(L / S)
    if G == 1:
        L_min = L_max = _averaged_num_layers
    else:
        L_min = max(_averaged_num_layers - L_OFFSET, 1)
        L_max = min(_averaged_num_layers + L_OFFSET, L - S + 1)
    L_candidates = list(range(L_min, L_max + 1))

    # Preprocess stable and delta costs
    (
        policy_choices,
        t_stable_processed,
        t_delta_processed,
        (selected_ckpt, selected_policy),
    ) = preprocess_stable_and_delta_costs(
        num_stages=num_stages,
        activation_checkpointing_tuning_enabled=activation_checkpointing_tuning_enabled,
        fixed_ckpt_value=fixed_ckpt_value,
        costs_stable=costs_stable,
        costs_delta=costs_delta,
        costs_stable_with_pre=costs_stable_with_pre,
        costs_delta_with_pre=costs_delta_with_pre,
        costs_stable_with_post=costs_stable_with_post,
        costs_delta_with_post=costs_delta_with_post,
        costs_stable_no_pp=costs_stable_no_pp,
        costs_delta_no_pp=costs_delta_no_pp,
    )

    model = pl.LpProblem("InterStageTuning", pl.LpMinimize)

    # 1. Decision Variables
    # l[i][j]: Whether stage i has j layers
    l = pl.LpVariable.dicts("l", (range(S), L_candidates), cat=pl.LpBinary)
    # z[i, j, k]: Whether stage i uses policy k and has j layers
    z = pl.LpVariable.dicts(
        "z",
        (
            (i, j, k)
            for i in range(S)
            for j in L_candidates
            for k in policy_choices[i][j]
        ),
        cat=pl.LpBinary,
    )

    # 2. Auxiliary Variables
    # 2.1 Stage stable latency and delta latency
    T = pl.LpVariable.dicts("T", range(S), lowBound=0, cat=pl.LpContinuous)
    D = pl.LpVariable.dicts("D", range(S), lowBound=0, cat=pl.LpContinuous)
    # 2.2. Max T and D
    Max_t = pl.LpVariable("Max_t", lowBound=0, cat=pl.LpContinuous)
    Max_d = pl.LpVariable("Max_d", lowBound=0, cat=pl.LpContinuous)

    # 3. Constraints
    # 3.1 Layer Constraint
    # Each stage can only choose one number of layers
    for i in range(S):
        model += pl.lpSum(l[i][j] for j in L_candidates) == 1
    # Sum of layers in each stage should be equal to the total number of layers
    model += pl.lpSum(l[i][j] * j for i in range(S) for j in L_candidates) == L

    # 3.2 Policy Constraint
    for i in range(S):
        model += (
            pl.lpSum(z[i, j, k] for j in L_candidates for k in policy_choices[i][j])
            == 1
        )

    # 3.3 Binary Var Constraint for z
    for i in range(S):
        for j in L_candidates:
            for k in policy_choices[i][j]:
                # Ensure z[i, j, k] is 1 only if l[i][j] is 1
                model += z[i, j, k] <= l[i][j]

    # 3.4 T and D Constraint
    for i in range(S):
        model += T[i] == pl.lpSum(
            t_stable_processed[i, j, k] * z[i, j, k]
            for j in L_candidates
            for k in policy_choices[i][j]
        )
        model += D[i] == pl.lpSum(
            t_delta_processed[i, j, k] * z[i, j, k]
            for j in L_candidates
            for k in policy_choices[i][j]
        )

    # 3.4 Max T and D Constraint
    for i in range(S):
        model += Max_t >= T[i]
        model += Max_d >= (D[i] - pl.lpSum(T[j] for j in range(i)))

    # 3.5 For faster convergence and early stopping
    model += Max_t * (G - 1) <= best_cost

    # 4. Objective
    model += Max_t * (G - 1) + pl.lpSum(T[i] for i in range(S)) + Max_d

    verbose = False
    time_limit = 600
    assert "PULP_CBC_CMD" in pl.listSolvers(onlyAvailable=True), (
        "Please install ILP solvers by 'sudo apt install coinor-cbc'")
    solver = pl.PULP_CBC_CMD(mip=True,
                            msg=verbose,
                            timeLimit=time_limit,
                            threads=multiprocessing.cpu_count())

    # 5. Solve
    try:
        model.solve(solver)
        # model.solve(pl.GLPK(msg=False))
    except Exception as e:
        logger.error(f"Failed to solve the MIP problem: {e}")
        return np.inf, None

    if pl.LpStatus[model.status] != "Optimal":
        return np.inf, None

    # Get the best cost and decision variables
    curr_best_cost = pl.value(model.objective)
    covered_layers = 0
    inter_solution = []
    for i in range(S):
        for j in L_candidates:
            for k in policy_choices[i][j]:
                if pl.value(z[i, j, k]) == 1:
                    # Get the ckpt layers and policy
                    curr_ckpt_layers = selected_ckpt[i][j][k]
                    curr_policy = selected_policy[i][j][k]
                    inter_solution.append(
                        (
                            (covered_layers, covered_layers + j - 1),
                            device_mesh,
                            curr_ckpt_layers,
                            curr_policy,
                        )
                    )
                    covered_layers += j
                    break

    logger.info(
        f"\n{S=}, {G=}, {device_mesh=}, {curr_best_cost=}: \n[Solution] {pformat(inter_solution)}"
    )

    return curr_best_cost, inter_solution


def preprocess_stable_and_delta_costs(
    num_stages: int,
    activation_checkpointing_tuning_enabled: bool,
    fixed_ckpt_value: Optional[int],
    costs_stable: np.ndarray,
    costs_delta: np.ndarray,
    costs_stable_with_pre: np.ndarray,
    costs_delta_with_pre: np.ndarray,
    costs_stable_with_post: np.ndarray,
    costs_delta_with_post: np.ndarray,
    costs_stable_no_pp: np.ndarray,
    costs_delta_no_pp: np.ndarray,
    output_sample_size: Optional[int] = None,
):
    S = num_stages
    _, L, C, P = costs_stable.shape
    assert (
        L == C - 1
    ), f"The number of layers and ckpt layer candidates should be the same. Got {L} and {C}."
    output_sample_size = output_sample_size or P

    # Raw costs
    t_stable = np.full((S, L + 1, L + 1, P), np.inf, dtype=np.float32)
    t_delta = np.full((S, L + 1, L + 1, P), np.inf, dtype=np.float32)

    # (1) without_pre_and_post
    for i in range(1, S - 1):
        t_stable[i, 1:, :, :] = costs_stable[S - i - 1]
        t_delta[i, 1:, :, :] = costs_delta[S - i - 1]
    # (2) with_pre
    t_stable[0, 1:, :, :] = costs_stable_with_pre[S - 1]
    t_delta[0, 1:, :, :] = costs_delta_with_pre[S - 1]
    # (3) with_post
    t_stable[S - 1, 1:, :, :] = costs_stable_with_post
    t_delta[S - 1, 1:, :, :] = costs_delta_with_post
    # (4) no_pp
    t_stable[0, -1, :, :] = costs_stable_no_pp
    t_delta[0, -1, :, :] = costs_delta_no_pp

    # Further processing
    if activation_checkpointing_tuning_enabled:
        t_stable_view = t_stable.reshape((S, L + 1, (L + 1) * P))
        t_delta_view = t_delta.reshape((S, L + 1, (L + 1) * P))
    elif fixed_ckpt_value is not None:
        # For each layer choice, select the same number of checkpoint choices
        t_stable_view = np.full((S, L + 1, P), np.inf, dtype=np.float32)
        t_delta_view = np.full((S, L + 1, P), np.inf, dtype=np.float32)
        for l in range(1, L + 1):
            t_stable_view[:, l, :] = t_stable[:, l, fixed_ckpt_value, :]
            t_delta_view[:, l, :] = t_delta[:, l, fixed_ckpt_value, :]
    else:
        # For each layer choice, select the same number of checkpoint choices
        t_stable_view = np.full((S, L + 1, P), np.inf, dtype=np.float32)
        t_delta_view = np.full((S, L + 1, P), np.inf, dtype=np.float32)
        for l in range(1, L + 1):
            t_stable_view[:, l, :] = t_stable[:, l, l, :]
            t_delta_view[:, l, :] = t_delta[:, l, l, :]

    t_stable_processed = np.full(
        (S, L + 1, output_sample_size), np.inf, dtype=np.float32
    )
    t_delta_processed = np.full(
        (S, L + 1, output_sample_size), np.inf, dtype=np.float32
    )
    selected_ckpt = np.full((S, L + 1, output_sample_size), -1, dtype=np.int32)
    selected_policy = np.full((S, L + 1, output_sample_size), -1, dtype=np.int32)
    unique_indices = []
    for i in range(S):
        indices, selected_t_stable, selected_t_delta = sample_pareto_frontier(
            t_stable_view[i],
            t_delta_view[i],
            sample_size=output_sample_size,
            alpha_based_sample_size=output_sample_size,
        )
        curr_unique_indices, selected_t_stable, selected_t_delta, _ = (
            fill_redundant_samples(selected_t_stable, selected_t_delta, indices)
        )
        t_stable_processed[i] = selected_t_stable
        t_delta_processed[i] = selected_t_delta
        unique_indices.append(curr_unique_indices)
        if activation_checkpointing_tuning_enabled:
            curr_selected_ckpt = indices // P
            curr_selected_policy = indices % P
        elif fixed_ckpt_value is not None:
            curr_selected_ckpt = np.broadcast_to(
                np.array([fixed_ckpt_value] * (L + 1)).reshape(-1, 1), (L + 1, output_sample_size)
            )
            curr_selected_policy = indices
        else:
            curr_selected_ckpt = np.broadcast_to(
                np.arange(L + 1).reshape(-1, 1), (L + 1, output_sample_size)
            )
            curr_selected_policy = indices

        selected_ckpt[i] = curr_selected_ckpt
        selected_policy[i] = curr_selected_policy

    t_stable_processed[t_stable_processed < 0] = 0
    t_delta_processed[t_delta_processed < 0] = 0
    t_stable_processed[t_stable_processed == np.inf] = MAX_VALUE
    t_delta_processed[t_delta_processed == np.inf] = MAX_VALUE

    return (
        unique_indices,
        t_stable_processed,
        t_delta_processed,
        (selected_ckpt, selected_policy),
    )
