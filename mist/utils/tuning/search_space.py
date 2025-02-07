from typing import Any, Dict, List, Tuple, Optional, Union
from functools import partial

import torch
import torch.nn as nn
import torch.fx as fx

from mist.config import MistConfig
from mist.analyzer.recorder import ExecInfoRecorder, ExecType
from mist.analyzer.layer_analyzer import LayerInfo
from mist.analyzer.strategy import create_decision_var_for_a_layer

# def build_optimization_problem_for_non_pp(
#         num_block_layers: int,
#         block_layer_info: LayerInfo,
#         config: MistConfig,
#         pre_layer_info: LayerInfo = None,
#         post_layer_info: LayerInfo = None,
#         warmup_cooldown_phases: int = 0,
# ):
#     # Create the decision variables
#     unique_layer_decision_vars = []
#     constraints = []

#     if has_pre_layer:
#         ...

#     if has_post_layer:
#         ...

#     for ...:
#         var, ..., constraint = create_decision_vars_for_a_stage_module

#     # Add the memory constraints
#     # should consider warmup and cooldown phases's saved_tensors
#     NotImplementedError

#     # Create the objective function: the latency for 1F1B

#     # Create tuner and optimization problem
#     # e.g. Hyperopt, Hyperactive, Nevergrad, etc.
#     algo = config.intra_op_tuner
#     tuner = build_tuner(algo, objective, constraints, unique_layer_decision_vars)
#     optim_prob = OptimizationProblem(...)

#     return optim_prob


# def tune_pp_layer_partitions(
#         num_stages: int,
#         num_layers: int,
#         partial_states: int,
#         saved_tensors: int,
# ):
#     """
#     Tune pp layer partitions based on the middle stage's 1B in 1F1B.
#     min max i=1 to s, l_i * partial_states + l_i * warmup_i * saved_tensors

#     Can be easily soved by an ILP solver.
#     """


#     NotImplementedError("TODO")

#     return layer_partition,


def build_optimization_problem_for_pp_fixed(
    block_layer_partition: List[int],
    block_layer_info: LayerInfo,
    gradient_accumulation_steps: int,
    config: MistConfig,
    pre_layer_info: LayerInfo = None,
    post_layer_info: LayerInfo = None,
):
    assert sum(block_layer_partition) == config.num_layers

    num_gpus = config.num_nodes * config.num_gpus_per_node
    num_stages = len(block_layer_partition)
    num_block_layers_per_stage = block_layer_partition
    assert num_gpus % num_stages == 0, "num_gpus must be divisible by num_stages"
    assert config.global_batch_size % gradient_accumulation_steps == 0
    batch_size_per_micro_batch = config.global_batch_size // gradient_accumulation_steps

    if config.num_nodes > num_stages:
        assert config.num_nodes % num_stages == 0
        num_nodes = config.num_nodes // num_stages
        num_gpus_per_node = config.num_gpus_per_node
    else:
        num_nodes = 1
        assert num_stages % config.num_nodes == 0
        assert config.num_gpus_per_node % (num_stages // config.num_nodes) == 0
        num_gpus_per_node = config.num_gpus_per_node // (num_stages // config.num_nodes)
    _create_decision_var_for_a_layer = partial(
        create_decision_var_for_a_layer,
        config=config,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        batch_size_per_micro_batch=batch_size_per_micro_batch,
    )
    _create_decision_var_for_block_layer = partial(
        _create_decision_var_for_a_layer, layer_info=block_layer_info
    )

    # Create the decision variables
    unique_layer_decision_vars = []
    constraints = []

    # Pre- and post- layers
    if pre_layer_info is not None:
        if config.strategy_granularity in {"model", "stage", "layer"}:
            pre_var = _create_decision_var_for_a_layer(
                "pre_layer", layer_info=pre_layer_info
            )
            unique_layer_decision_vars.append(pre_var)
        elif config.strategy_granularity in {"micro_batch", "phase"}:
            # Pre-layer can only show up in the first stage
            # which should have (num_warmup_phases + 1 (1F1B)) phases
            for i in range(num_stages):
                pre_var = _create_decision_var_for_a_layer(
                    f"pre_layer_mb{i}", layer_info=pre_layer_info
                )
                unique_layer_decision_vars.append(pre_var)
        else:
            raise ValueError(
                f"Unknown strategy_granularity: {config.strategy_granularity}"
            )

    if post_layer_info is not None:
        if config.strategy_granularity in {"model", "stage", "layer"}:
            post_var = _create_decision_var_for_a_layer(
                "post_layer", layer_info=post_layer_info
            )
            unique_layer_decision_vars.append(post_var)
        elif config.strategy_granularity in {"micro_batch", "phase"}:
            # Post-layer can only show up in the last stage
            # which should have 1 (1F1B) phase
            post_var = _create_decision_var_for_a_layer(
                "post_layer", layer_info=post_layer_info
            )
            unique_layer_decision_vars.append(post_var)
        else:
            raise ValueError(
                f"Unknown strategy_granularity: {config.strategy_granularity}"
            )

    # Create the decision variables for block layers
    if config.strategy_granularity == "model":
        # All phases share the same strategy
        block_var = _create_decision_var_for_block_layer("block_layer")
        unique_layer_decision_vars.append(block_var)
        raise NotImplementedError("TODO: build the optimization problem")
    elif config.strategy_granularity == "stage":
        # All phases in a stage share the same strategy
        for stage_idx in range(num_stages):
            block_var = _create_decision_var_for_block_layer(
                f"block_layer_stage{stage_idx}"
            )
            unique_layer_decision_vars.append(block_var)
        raise NotImplementedError(f"TODO: build the optimization problem")
    elif config.strategy_granularity == "micro_batch":
        decision_vars = []
        block_var = _create_decision_var_for_block_layer("block_layer")
        for stage_idx in range(num_stages):
            warmup, fb = calculate_num_warmup_and_1f1b_phases(
                stage_idx, num_stages, gradient_accumulation_steps
            )
            stage_decision_vars = []
            for mb_idx in range(warmup + 1):
                stage_decision_vars.append(block_var)
            decision_vars.append(stage_decision_vars)

        # Create the objective and memory constraints
        def objective(trial):
            layer_infos = []
            # Create symbols for the decision variables
            for var in decision_vars:
                data_point = []
                for name, choices in var.search_space.items():
                    value = trial.suggest_categorical(name, choices)
                    data_point.append(value)
                layer_info = var.data_point_to_layer_info[data_point]
                layer_infos.append(layer_info)

            # Memory constraints
            peak_memory = 0

        latency = calculate_pipe_latency(
            stage_to_micro_batch_to_latency,
        )

        memory_term = 0
        for stage in stages:
            peak = compute_peak_memory(stage)
            if peak > config.memory_capacity:
                memory_term += (peak - config.memory_capacity) ** 2

        return latency + memory_term

        raise NotImplementedError("TODO: Add memory constraints")
    elif config.strategy_granularity == "layer":
        for stage_idx in range(num_stages):
            for layer_idx in range(num_block_layers_per_stage[stage_idx]):
                block_var = _create_decision_var_for_block_layer(
                    f"block_layer_stage{stage_idx}_layer{layer_idx}"
                )
                unique_layer_decision_vars.append(block_var)
        raise NotImplementedError(f"TODO: build the optimization problem")
    elif config.strategy_granularity == "phase":
        for stage_idx in range(num_stages):
            warmup, fb = calculate_num_warmup_and_1f1b_phases(
                stage_idx, num_stages, gradient_accumulation_steps
            )
            for mb_idx in range(warmup + 1):
                for layer_idx in range(num_block_layers_per_stage[stage_idx]):
                    block_var = _create_decision_var_for_block_layer(
                        f"block_layer_stage{stage_idx}_layer{layer_idx}_mb{mb_idx}",
                    )
                    unique_layer_decision_vars.append(block_var)
        raise NotImplementedError(f"TODO: build the optimization problem")
    else:
        raise ValueError(f"Unknown strategy_granularity: {config.strategy_granularity}")

    # Add the memory constraints
    NotImplementedError("TODO: add memory constraints")

    # Create the objective function: the latency for the whole pipeline

    # Create tuner and optimization problem
    algo = config.inter_op_tuner
    tuner = build_tuner(algo, objective, constraints, unique_layer_decision_vars)
    optim_prob = OptimizationProblem(...)

    return optim_prob


def build_optimization_problem(
    layer_infos: Dict[str, Dict[Tuple[ExecType, bool], LayerInfo]],
    config: MistConfig,
):
    """Build optimization problem for the given layer infos.

    Parameters
    ----------
    layer_infos
        layer_name -> (exec_type, is_ckpt) -> layer_info
    config
        the mist configuration
    """

    # Construct the pp search space if needed
    if not config.pp_enabled:
        # optim_prob = build_optimization_problem_for_non_pp(
        #     ..., pre_layer=True, post_layer=True
        # )
        # results = optim_prob.tune()
        pass
    else:
        if config.pp_tuning_enabled:
            best_optim_prob = None
            best_latency = float("inf")
            # 1. tune the search space based on the middle stage
            # 2. min max i=1 to s, l_i * partial_states + l_i * warmup_i * saved_tensors
            # 3. tune the search space based on the min max
            # for num_stage in config.num_stages_candidates:
            #     surrogate_optim_prob = build_optimization_problem_for_non_pp(...)
            #     surrogate_results = optim_prob.tune()
            #     layer_partition, surrogate_latency = tune_pp_layer_partitions(...)
            #     optim_prob = build_optimization_problem_for_pp_fixed(...)
            #     resuls = optim_prob.tune()
            #     latency = get_latency_from_results()
            #     if latency < best_latency:
            #         best_optim_prob = optim_prob
            #         best_latency = latency
        else:
            num_stages = config.num_stages_if_tuning_disabled
            block_layer_partition = _uniform_layer_partition(
                config.num_layers, num_stages
            )
            optim_prob = build_optimization_problem_for_pp_fixed(
                block_layer_partition,
                block_layer_info=layer_infos["block_layer"],
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                config=config,
                pre_layer_info=layer_infos["pre_layer"],
                post_layer_info=layer_infos["post_layer"],
            )
            results = optim_prob.tune()

    # get_layer_strategy_from_results()
    # save_and_draw_pp_results_if_needed()
    # dump_to_run_config_if_needed()

    return optim_prob, results


def _uniform_layer_partition(num_layers: int, num_stages: int):
    """Uniformly partition the layers into stages."""
    num_layers_per_stage = num_layers // num_stages
    layer_partition = [num_layers_per_stage] * num_stages
    # Add layers to the last few stages
    for i in range(num_layers % num_stages):
        layer_partition[-i - 1] += 1
    return layer_partition


def calculate_num_warmup_and_1f1b_phases(
    stage_idx, num_stages, gradient_accumulation_steps
):
    warmup = num_stages - stage_idx - 1
    fb = gradient_accumulation_steps - warmup
    return warmup, fb
