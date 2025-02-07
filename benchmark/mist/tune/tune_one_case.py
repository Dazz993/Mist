import hydra
import os
import pickle
import tqdm
import contextlib
from copy import deepcopy
from functools import partial
from omegaconf import DictConfig, OmegaConf
from pprint import pprint, pformat
from typing import List, Optional, Union, Tuple, Any, Dict, Callable

import numpy as np
import torch
import torch.nn as nn

from mist import global_symbol_manager as gsm
from mist.analyzer.layer_analyzer import LayerInfo, analyze_blocks
from mist.analyzer.batched_module_analyzer import batched_stage_analyze, OFFLOADING_RATIO_CANDIDATES
from mist.config import (
    MistConfig,
    ModelConfig,
    TrainingConfig,
    HardwareConfig,
    StrategyConfig,
)
from mist.distributed.overrides import MistProcessGroup
from mist.analyzer.intra_tuning import (
    IntraStageTuner,
    intra_stage_tune,
)
from mist.analyzer.inter_tuning import inter_stage_tune
from mist.analyzer.inter_tuning_uniform import (
    inter_stage_tune as inter_stage_tune_uniform,
)
from mist.analyzer.inter_tuning_uniform_device import (
    inter_stage_tune as inter_stage_tune_uniform_device,
)
from mist.analyzer.inter_tuning_mip import inter_stage_tune as inter_stage_tune_mip
from mist.logger import get_logger
from mist.pipeline_parallel.pipe_module import build_pipe_modules_for_analyzing
from mist.tracer.symbolic_tracer import mist_trace
from mist.tuning.optimization import (
    _get_device_mesh_candidates,
    _get_grad_accumu_steps_candidates,
    build_and_tune_optimization_problem,
)
from mist.utils.common import load_pickle, save_pickle, load_json, save_json
from mist.utils.memory import cuda_empty_cache
from mist.utils.device import mock_cuda_device_name_if_needed
from mist.symbols import temporarily_set_sp_eq_ne
from mist.benchmark.common import get_common_providers_for_analysis_and_tuning


logger = get_logger(__name__)
logger.setLevel("DEBUG")

DISABLE_TQDM = False
if str(os.getenv("DISABLE_TQDM", False)).lower() == "true":
    DISABLE_TQDM = True


def best_solution_yaml(cfg: DictConfig, best_solution: List[Any]):
    assert len(best_solution) == 2, (
        f"len(best_solution)={len(best_solution)}. " f"best_solution={best_solution}."
    )
    gradient_accumulation_steps, stage_solutions = best_solution
    layer_partitions = []
    device_assignment = []
    gradient_checkpointing = []
    policy_indices = []
    stage_strategies = []

    for stage_idx, stage_solution in enumerate(stage_solutions):
        (layer_partition, device_mesh, num_ckpt, policy_index), stage_strategy = stage_solution
        layer_start, layer_end = layer_partition
        layer_partitions.append(int(layer_end - layer_start + 1))
        device_assignment.append([int(x) for x in device_mesh])
        gradient_checkpointing.append(int(num_ckpt))
        policy_indices.append(int(policy_index))
        stage_strategies.append(list(stage_strategy))

    # Update the pre-post strategy array with the strategy of the last stage
    pre_post_strategy_array = [None, None, None, 0, 0, 1, 0, 0, 0, 0]
    # micro_batch_size, dp_size, tp_size
    pre_post_strategy_array[0] = stage_strategies[-1][0]
    pre_post_strategy_array[1] = stage_strategies[-1][1]
    pre_post_strategy_array[2] = stage_strategies[-1][2]

    # Update the config with the best solution
    cfg.strategy.enabled = True
    cfg.strategy.gradient_accumulation_steps = gradient_accumulation_steps
    cfg.strategy.layer_partitions = layer_partitions
    cfg.strategy.device_assignment = device_assignment
    cfg.strategy.gradient_checkpointing = gradient_checkpointing
    cfg.strategy.stage_strategies = stage_strategies
    cfg.strategy.pre_post_strategy = "preset"
    cfg.strategy.pre_post_strategy_array = pre_post_strategy_array

    # Save the best solution to a yaml file
    best_solution_yaml = OmegaConf.to_yaml(cfg)
    return best_solution_yaml


def tune(
    mist_config: MistConfig,
):
    model_config = mist_config.model
    training_config = mist_config.training
    strategy_config = mist_config.strategy
    hardware_config = mist_config.hardware
    tuning_config = mist_config.tuning
    model_config.tensor_parallel = True

    data = get_common_providers_for_analysis_and_tuning(
        mist_config=mist_config,
        num_hidden_layers=1,
        force_rebuild=False,
    )
    layer_infos = data["layer_infos"]

    tuning_granularity = tuning_config.tuning_granularity

    # if tuning_config.tuning_granularity == "uniform-pp":
    #     tuning_granularity = "uniform-device-pp-mip"
    #     tuning_config.layers_offset = 0

    if tuning_config.tuning_granularity == "no-pp":
        num_nodes = hardware_config.num_nodes
        num_gpus_per_node = hardware_config.num_gpus_per_node
        device_mesh_candidates = [(num_nodes, num_gpus_per_node)]
        grad_accumu_steps_candidates = _get_grad_accumu_steps_candidates(
            global_batch_size=training_config.global_batch_size,
        )
        # Intra-stage strategy tuning
        intra_stage_results = intra_stage_tune(
            block_layer_info=layer_infos["block_layer"],
            pre_layer_info=layer_infos["pre_layer"],
            post_layer_info=layer_infos["post_layer"],
            num_layers=model_config.num_hidden_layers,
            device_mesh_candidates=device_mesh_candidates,
            gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
            mist_config=mist_config,
            disable_tqdm=DISABLE_TQDM,
            force_recompute=False,
        )
        # Get the costs and scale them with the gradient accumulation steps
        if tuning_config.activation_checkpointing_tuning_enabled:
            costs = intra_stage_results.costs_no_pp
            solutions = intra_stage_results.solutions_no_pp
        else:
            # FIXME(zhanda): this is wrong. It is not -1, it should be
            # the last index of the number of layers
            costs = intra_stage_results.costs_no_pp[..., -1]
            solutions = intra_stage_results.solutions_no_pp[..., -1, :]
        for i, grad_accumu_step in enumerate(grad_accumu_steps_candidates):
            costs[i] *= grad_accumu_step
        best_cost = np.min(costs)
        best_solution_idx = np.unravel_index(np.argmin(costs), costs.shape)
        best_solution = solutions[best_solution_idx].tolist()
        best_grad_accumu_steps = grad_accumu_steps_candidates[best_solution_idx[0]]
        best_device_mesh = device_mesh_candidates[best_solution_idx[1]]
        if tuning_config.activation_checkpointing_tuning_enabled:
            best_activation_checkpointing_layers = best_solution_idx[2]
        else:
            best_activation_checkpointing_layers = model_config.num_hidden_layers
        # Update the best solution to match the standard format
        best_intra_solution = deepcopy(best_solution)
        best_intra_solution[:6] = [int(x) for x in best_intra_solution[:6]]
        best_intra_solution[6:] = [float(x) for x in best_intra_solution[6:]]
        best_intra_solution = tuple(best_intra_solution)
        best_inter_solution = (
            (0, model_config.num_hidden_layers - 1),
            best_device_mesh,
            best_activation_checkpointing_layers,
        )
        best_solution = [
            best_grad_accumu_steps,
            [(best_inter_solution, best_intra_solution)],
        ]
        logger.info(f"Best cost: {best_cost}")
        logger.info(f"Best grad_accumu_steps: {best_grad_accumu_steps}")
        logger.info(f"Best device_mesh: {best_device_mesh}")
        logger.info(
            f"Best activation_checkpointing_layers: {best_activation_checkpointing_layers}"
        )
        logger.info(f"Best solution: {pformat(best_solution)}")

    elif tuning_config.tuning_granularity == "uniform-pp":
        # Deprecated: See the transformation above
        # Get the device mesh and gradient accumulation steps candidates
        device_mesh_candidates = _get_device_mesh_candidates(
            num_nodes=hardware_config.num_nodes,
            num_gpus_per_node=hardware_config.num_gpus_per_node,
            contiguous_inter_node=False,
        )
        grad_accumu_steps_candidates = _get_grad_accumu_steps_candidates(
            global_batch_size=training_config.global_batch_size,
        )
        if getattr(mist_config, "disable_tp_tuning", False):
            tp_candidates = [1]
        else:
            tp_candidates = [1, 2, 4, 8]
            tp_candidates = [tp for tp in tp_candidates if tp <= hardware_config.num_gpus_per_node]
        best_cost = float("inf")
        best_solution = None
        for tp_size in tp_candidates:
            # Intra-stage strategy tuning
            intra_stage_results = intra_stage_tune(
                block_layer_info=layer_infos["block_layer"],
                pre_layer_info=layer_infos["pre_layer"],
                post_layer_info=layer_infos["post_layer"],
                num_layers=model_config.num_hidden_layers,
                device_mesh_candidates=device_mesh_candidates,
                gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
                tp_size=tp_size,
                mist_config=mist_config,
                disable_tqdm=DISABLE_TQDM,
                force_recompute=False,
            )
            # Inter-stage strategy tuning
            curr_best_cost, curr_best_solution = inter_stage_tune_uniform(
                num_layers=model_config.num_hidden_layers,
                num_nodes=hardware_config.num_nodes,
                num_gpus_per_node=hardware_config.num_gpus_per_node,
                # device_mesh_candidates=device_mesh_candidates,
                gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
                intra_stage_results=intra_stage_results,
                config=mist_config,
            )
            if curr_best_cost < best_cost:
                best_cost = curr_best_cost
                best_solution = curr_best_solution
        logger.info(f"Best cost: {best_cost}")
        logger.info(f"Best solution: {pformat(best_solution)}")
    
    elif tuning_config.tuning_granularity == "uniform-pp-simple-heuristic-mem-opt":
        # Deprecated: See the transformation above
        # Get the device mesh and gradient accumulation steps candidates
        device_mesh_candidates = _get_device_mesh_candidates(
            num_nodes=hardware_config.num_nodes,
            num_gpus_per_node=hardware_config.num_gpus_per_node,
            contiguous_inter_node=False,
        )
        grad_accumu_steps_candidates = _get_grad_accumu_steps_candidates(
            global_batch_size=training_config.global_batch_size,
        )
        if getattr(mist_config, "disable_tp_tuning", False):
            tp_candidates = [1]
        else:
            tp_candidates = [1, 2, 4, 8]
            tp_candidates = [tp for tp in tp_candidates if tp <= hardware_config.num_gpus_per_node]
        best_cost = float("inf")
        best_solution = None
        for ao_ratio in OFFLOADING_RATIO_CANDIDATES:
            # mist_config.fixed_ao_ratio = ao_ratio
            mist_config.fixed_oo_ratio = ao_ratio
            mist_config.tuning.activation_offloading_enabled = False
            for tp_size in tp_candidates:
                # Intra-stage strategy tuning
                intra_stage_results = intra_stage_tune(
                    block_layer_info=layer_infos["block_layer"],
                    pre_layer_info=layer_infos["pre_layer"],
                    post_layer_info=layer_infos["post_layer"],
                    num_layers=model_config.num_hidden_layers,
                    device_mesh_candidates=device_mesh_candidates,
                    gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
                    tp_size=tp_size,
                    mist_config=mist_config,
                    disable_tqdm=DISABLE_TQDM,
                    force_recompute=False,
                )
                # Inter-stage strategy tuning
                curr_best_cost, curr_best_solution = inter_stage_tune_uniform(
                    num_layers=model_config.num_hidden_layers,
                    num_nodes=hardware_config.num_nodes,
                    num_gpus_per_node=hardware_config.num_gpus_per_node,
                    # device_mesh_candidates=device_mesh_candidates,
                    gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
                    intra_stage_results=intra_stage_results,
                    config=mist_config,
                )
                if curr_best_cost < best_cost:
                    best_cost = curr_best_cost
                    best_solution = curr_best_solution
                
            # logger.info(f"AO ratio: {ao_ratio} - TP size: {tp_size}")
            # logger.info(f"Best cost: {best_cost}")
            # logger.info(f"Best solution: {pformat(best_solution)}")
        logger.info(f"Best cost: {best_cost}")
        logger.info(f"Best solution: {pformat(best_solution)}")

    elif tuning_config.tuning_granularity == "uniform-device-pp":
        # Get the device mesh and gradient accumulation steps candidates
        device_mesh_candidates = _get_device_mesh_candidates(
            num_nodes=hardware_config.num_nodes,
            num_gpus_per_node=hardware_config.num_gpus_per_node,
            contiguous_inter_node=False,
        )
        grad_accumu_steps_candidates = _get_grad_accumu_steps_candidates(
            global_batch_size=training_config.global_batch_size,
        )
        # Intra-stage strategy tuning
        intra_stage_results = intra_stage_tune(
            block_layer_info=layer_infos["block_layer"],
            pre_layer_info=layer_infos["pre_layer"],
            post_layer_info=layer_infos["post_layer"],
            num_layers=model_config.num_hidden_layers,
            device_mesh_candidates=device_mesh_candidates,
            gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
            mist_config=mist_config,
            disable_tqdm=DISABLE_TQDM,
            force_recompute=False,
        )
        # Inter-stage strategy tuning
        best_cost, best_solution = inter_stage_tune_uniform_device(
            num_layers=model_config.num_hidden_layers,
            num_nodes=hardware_config.num_nodes,
            num_gpus_per_node=hardware_config.num_gpus_per_node,
            device_mesh_candidates=device_mesh_candidates,
            gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
            intra_stage_results=intra_stage_results,
            config=mist_config,
        )
        logger.info(f"Best cost: {best_cost}")
        logger.info(f"Best solution: {pformat(best_solution)}")

    elif tuning_config.tuning_granularity == "inter-stage":
        # Get the device mesh and gradient accumulation steps candidates
        device_mesh_candidates = _get_device_mesh_candidates(
            num_nodes=hardware_config.num_nodes,
            num_gpus_per_node=hardware_config.num_gpus_per_node,
            contiguous_inter_node=False,
        )
        grad_accumu_steps_candidates = _get_grad_accumu_steps_candidates(
            global_batch_size=training_config.global_batch_size,
        )
        # Intra-stage strategy tuning
        intra_stage_results = intra_stage_tune(
            block_layer_info=layer_infos["block_layer"],
            pre_layer_info=layer_infos["pre_layer"],
            post_layer_info=layer_infos["post_layer"],
            num_layers=model_config.num_hidden_layers,
            device_mesh_candidates=device_mesh_candidates,
            gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
            mist_config=mist_config,
            disable_tqdm=DISABLE_TQDM,
            force_recompute=False,
        )
        # Inter-stage strategy tuning
        best_cost, best_solution = inter_stage_tune(
            num_layers=model_config.num_hidden_layers,
            num_devices=hardware_config.num_nodes * hardware_config.num_gpus_per_node,
            device_mesh_candidates=device_mesh_candidates,
            gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
            intra_stage_results=intra_stage_results,
        )
        logger.info(f"Best cost: {best_cost}")
        logger.info(f"Best solution: {pformat(best_solution)}")

    elif tuning_config.tuning_granularity == "uniform-device-pp-mip":
        # Get the device mesh and gradient accumulation steps candidates
        device_mesh_candidates = _get_device_mesh_candidates(
            num_nodes=hardware_config.num_nodes,
            num_gpus_per_node=hardware_config.num_gpus_per_node,
            contiguous_inter_node=False,
        )
        grad_accumu_steps_candidates = _get_grad_accumu_steps_candidates(
            global_batch_size=training_config.global_batch_size,
        )
        if getattr(mist_config, "disable_tp_tuning", False):
            tp_candidates = [1]
        else:
            tp_candidates = [1, 2, 4, 8]
            tp_candidates = [tp for tp in tp_candidates if tp <= hardware_config.num_gpus_per_node]

        best_cost = float("inf")
        best_solution = None
        for tp_size in tp_candidates:
            # Intra-stage strategy tuning
            intra_stage_results = intra_stage_tune(
                block_layer_info=layer_infos["block_layer"],
                pre_layer_info=layer_infos["pre_layer"],
                post_layer_info=layer_infos["post_layer"],
                num_layers=model_config.num_hidden_layers,
                device_mesh_candidates=device_mesh_candidates,
                gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
                tp_size=tp_size,
                mist_config=mist_config,
                disable_tqdm=DISABLE_TQDM,
                force_recompute=False,
                sample_size=tuning_config.sample_size,
            )
            # Inter-stage strategy tuning
            curr_best_cost, curr_best_solution = inter_stage_tune_mip(
                num_layers=model_config.num_hidden_layers,
                num_nodes=hardware_config.num_nodes,
                num_gpus_per_node=hardware_config.num_gpus_per_node,
                device_mesh_candidates=device_mesh_candidates,
                gradient_accumulation_steps_candidates=grad_accumu_steps_candidates,
                intra_stage_results=intra_stage_results,
                config=mist_config,
            )
            if curr_best_cost < best_cost:
                best_cost = curr_best_cost
                best_solution = curr_best_solution

        logger.info(f"Best cost: {best_cost}")
        logger.info(f"Best solution: {pformat(best_solution)}")

    else:
        raise ValueError(
            f"Unknown tuning granularity: {tuning_config.tuning_granularity}"
        )

    return best_cost, best_solution


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config.yaml",
)
def main(cfg: DictConfig) -> None:
    cfg.strategy.enabled = False
    cfg.tuning.enabled = True
    mist_config = MistConfig.from_dict_config(cfg)
    best_cost, best_solution = tune(mist_config)
    best_solution_yaml_str = best_solution_yaml(cfg, best_solution)

    # Output the best solution if the output path is specified
    if getattr(cfg, "output_path", None) is not None:
        folder_path = os.path.dirname(cfg.output_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not cfg.output_path.endswith(".yaml"):
            cfg.output_path += ".yaml"
        config_file_name = os.path.basename(cfg.output_path).split(".")[0]
        # Save the best solution to a yaml file
        with open(cfg.output_path, "w") as f:
            f.write(best_solution_yaml_str)
        # Save the best cost and best solution to a json file
        summary_path = os.path.join(folder_path, "summary.json")
        data = load_json(summary_path) if os.path.exists(summary_path) else {}
        key = config_file_name
        value = {
            "tuning_best_cost": f"{best_cost:.4f}",
            "tuning_best_solution": pformat(best_solution),
        }
        data.setdefault(key, {}).update(value)
        save_json(data, summary_path)
        logger.info(f"Saved the best solution to {cfg.output_path}")


if __name__ == "__main__":
    with mock_cuda_device_name_if_needed():
        main()
