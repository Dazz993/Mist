import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
from pprint import pprint, pformat
from typing import List, Optional, Union, Tuple, Any, Dict, Callable

import torch
import torch.nn as nn

from mist import global_symbol_manager as gsm
from mist.analyzer.layer_analyzer import LayerInfo, analyze_blocks
from mist.analyzer.batched_module_analyzer import batched_stage_analyze
from mist.config import (
    MistConfig,
    ModelConfig,
    TrainingConfig,
    HardwareConfig,
    StrategyConfig,
)
from mist.distributed.overrides import MistProcessGroup
from mist.logger import get_logger
from mist.pipeline_parallel.pipe_module import build_pipe_modules_for_analyzing
from mist.tracer.symbolic_tracer import mist_trace
from mist.tuning.optimization import build_and_tune_optimization_problem
from mist.symbols import temporarily_set_sp_eq_ne
from mist.utils.memory import cuda_empty_cache
from mist.utils.device import get_device, mock_cuda_device_name_if_needed
from mist.benchmark.common import get_common_providers_for_analysis_and_tuning
from mist.utils.common import load_json, save_json


logger = get_logger(__name__)
logger.setLevel("DEBUG")


def analyze(
    mist_config: MistConfig,
):
    model_config = mist_config.model
    training_config = mist_config.training
    strategy_config = mist_config.strategy
    hardware_config = mist_config.hardware
    model_config.tensor_parallel = True

    # Hyperparameters
    num_stages = strategy_config.num_stages
    gradient_accumulation_steps = strategy_config.gradient_accumulation_steps

    data = get_common_providers_for_analysis_and_tuning(
        mist_config=mist_config,
        num_hidden_layers=1,
        force_rebuild=False,
    )
    layer_infos = data["layer_infos"]

    def get_stage_results(stage_idx: int, use_pre: bool, use_post: bool):
        pre_saved_micro_batches = min(
            num_stages - stage_idx - 1, gradient_accumulation_steps - 1
        )
        curr_num_layers = strategy_config.layer_partitions[stage_idx]
        num_nodes, num_gpus_per_node = strategy_config.device_assignment[stage_idx]
        num_ckpt_layers = strategy_config.gradient_checkpointing[stage_idx]
        stage_strategies = strategy_config.stage_strategies[stage_idx]
        (
            results,
            results_with_pre,
            results_with_post,
            results_with_pre_and_post,
        ) = batched_stage_analyze(
            block_layer_info=layer_infos["block_layer"],
            pre_layer_info=layer_infos["pre_layer"],
            post_layer_info=layer_infos["post_layer"],
            pre_saved_micro_batches_candidates=[pre_saved_micro_batches],
            num_layers_candidates=[curr_num_layers],
            num_ckpt_layers_candidates=[num_ckpt_layers],
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            gradient_accumulation_steps=gradient_accumulation_steps,
            stage_strategies=stage_strategies,
            config=mist_config,
        )
        sample = (pre_saved_micro_batches, curr_num_layers, num_ckpt_layers)
        if use_pre and use_post:
            return results_with_pre_and_post[sample]
        elif use_pre:
            return results_with_pre[sample]
        elif use_post:
            return results_with_post[sample]
        else:
            return results[sample]

    def get_stable_and_delta_latency(
        result: Dict[str, Any], use_pre: bool, use_post: bool
    ):
        if use_pre and use_post:
            stable_latency = result["****latency_stable_with_pre_and_post"]
            delta_latency = result["****latency_delta_with_pre_and_post"]
        elif use_pre:
            stable_latency = result["****latency_stable_with_pre"]
            delta_latency = result["****latency_delta_with_pre"]
        elif use_post:
            stable_latency = result["****latency_stable_with_post"]
            delta_latency = result["****latency_delta_with_post"]
        else:
            stable_latency = result["****latency_stable"]
            delta_latency = result["****latency_delta"]
        return stable_latency, delta_latency

    # Delta latency for the first stage, only used when num_stages > 1 and enable_non_uniform_pp_micro_batch_analysis is True
    stage_stable_latencies = []
    stage_delta_latencies = []

    if num_stages == 1:
        result = get_stage_results(stage_idx=0, use_pre=True, use_post=True)
        logger.info(pformat(result))
        # Formatting the result
        total_latency = result["latency"].item() * gradient_accumulation_steps
        stage_stable_latencies = [result["latency"].item()]
        stage_stable_latencies = [0]
        stage_peak_fwd_memories = [result["mem_fwd_peak"].item()]
        stage_peak_bwd_memories = [result["mem_bwd_peak"].item()]
        logger.info(f"Total latency: {total_latency}")
        logger.info(f"Peak fwd memory: {stage_peak_fwd_memories}")
        logger.info(f"Peak bwd memory: {stage_peak_bwd_memories}")
    elif getattr(mist_config, "disable_non_uniform_pp_micro_batch_analysis", False):
        stage_results = []
        stage_peak_fwd_memories = []
        stage_peak_bwd_memories = []
        for stage_idx in range(num_stages):
            use_pre = stage_idx == 0
            use_post = stage_idx == num_stages - 1
            result = get_stage_results(
                stage_idx=stage_idx, use_pre=use_pre, use_post=use_post
            )
            stage_results.append(result)
            stage_stable_latencies.append(result["latency"].item())
            stage_delta_latencies.append(0)
            stage_peak_fwd_memories.append(result["mem_fwd_peak"].item())
            stage_peak_bwd_memories.append(result["mem_bwd_peak"].item())
            logger.info(f"Stage {stage_idx}\n{pformat(result)}")
        total_latency = sum(stage_stable_latencies) + max(stage_stable_latencies) * (
            gradient_accumulation_steps - 1
        )
        logger.info(f"Stage Latencies: {stage_stable_latencies}")
        logger.info(f"Total latency: {total_latency}")
        logger.info(f"Peak fwd memory: {stage_peak_fwd_memories}")
        logger.info(f"Peak bwd memory: {stage_peak_bwd_memories}")
    else:
        stage_results = []
        stage_peak_fwd_memories = []
        stage_peak_bwd_memories = []
        for stage_idx in range(num_stages):
            use_pre = stage_idx == 0
            use_post = stage_idx == num_stages - 1
            result = get_stage_results(
                stage_idx=stage_idx, use_pre=use_pre, use_post=use_post
            )
            stable_latency, delta_latency = get_stable_and_delta_latency(
                result, use_pre, use_post
            )
            stage_results.append(result)
            stage_stable_latencies.append(stable_latency)
            stage_delta_latencies.append(delta_latency)
            stage_peak_fwd_memories.append(result["mem_fwd_peak"].item())
            stage_peak_bwd_memories.append(result["mem_bwd_peak"].item())
            logger.info(f"Stage {stage_idx}\n{pformat(result)}")

        total_latency = (
            sum(stage_stable_latencies)
            + max(stage_stable_latencies) * (gradient_accumulation_steps - 1)
            + max(
                delta - sum(stage_stable_latencies[i] for i in range(s))
                for s, delta in enumerate(stage_delta_latencies)
            )
        )

        logger.info(f"Stage Stable Latencies: {stage_stable_latencies}")
        logger.info(f"Stage Delta Latencies: {stage_delta_latencies}")
        logger.info(f"Total latency: {total_latency}")
        logger.info(f"Peak fwd memory: {stage_peak_fwd_memories}")
        logger.info(f"Peak bwd memory: {stage_peak_bwd_memories}")

    stage_peak_memories = [
        max(fwd, bwd)
        for fwd, bwd in zip(stage_peak_fwd_memories, stage_peak_bwd_memories)
    ]

    return (
        total_latency,
        stage_stable_latencies,
        stage_delta_latencies,
        stage_peak_memories,
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config.yaml",
)
def main(cfg: DictConfig) -> None:
    # logger.info(OmegaConf.to_yaml(cfg))
    cfg.strategy.enabled = True
    cfg.tuning.enabled = False
    mist_config = MistConfig.from_dict_config(cfg)
    (
        total_latency,
        stage_stable_latencies,
        stage_delta_latencies,
        stage_peak_memories,
    ) = analyze(mist_config)

    # Output the total latency and peak memories if the output path is specified
    if getattr(cfg, "output_path", None) is not None:
        folder_path = os.path.dirname(cfg.output_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        config_file_name = Path(os.path.basename(cfg.output_path)).stem
        # Save the total latency and peak memories to a summary file
        summary_path = os.path.join(folder_path, "summary.json")
        data = load_json(summary_path) if os.path.exists(summary_path) else {}
        key = config_file_name
        stage_stable_latencies_str = ", ".join(
            [f"{lat:.4f}" for lat in stage_stable_latencies]
        )
        stage_stable_latencies_str = f"[{stage_stable_latencies_str}]"
        stage_delta_latencies_str = ", ".join(
            [f"{lat:.4f}" for lat in stage_delta_latencies]
        )
        stage_delta_latencies_str = f"[{stage_delta_latencies_str}]"
        stage_peak_memories_str = ", ".join(
            [f"{mem:.0f} MB" for mem in stage_peak_memories]
        )
        stage_peak_memories_str = f"[{stage_peak_memories_str}]"
        value = {
            "analyzed_total_cost": f"{total_latency:.4f}",
            "analyzed_stage_stable_latencies": stage_stable_latencies_str,
            "analyzed_stage_delta_latencies": stage_delta_latencies_str,
            "analyzed_stage_peak_memories": stage_peak_memories_str,
        }
        data.setdefault(key, {}).update(value)
        save_json(data, summary_path)


if __name__ == "__main__":
    with mock_cuda_device_name_if_needed():
        main()
