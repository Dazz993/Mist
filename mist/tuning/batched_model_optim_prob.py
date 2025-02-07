import itertools
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, cache
from time import perf_counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional, Iterator, Any, Sequence


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
from mist.tuning.optim_prob_base import OptimProb, _calculate_search_space_size
from mist.utils.pipeline_parallel import calculate_num_warmup_and_1f1b_phases
from mist.utils.sympy import autowrap_with_cython

logger = get_logger(__name__)

GB = 1024**3


def batched_exec_latency_per_layer(
    config: MistConfig, layer_info: LayerInfo, candidates: Sequence[Tuple[int, int]]
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """
    Get fwd and bwd latency for layer with batched execution.

    Parameters
    ----------
    config : MistConfig
        MistConfig object.
    layer_info : LayerInfo
        LayerInfo object.
    candidates : List[Tuple[int, int]]
        List of (batch_size, num_gpus) candidates.
    """
    fwd_latencies = {}
    bwd_latencies = {}

    for candidate in tqdm(candidates):
        fwd_latency, bwd_latency = exec_latency_per_layer(config, layer_info, candidate)
        fwd_latencies[candidate] = fwd_latency
        bwd_latencies[candidate] = bwd_latency

    return fwd_latencies, bwd_latencies


@cache
def exec_latency_per_layer(
    config: MistConfig, layer_info: LayerInfo, candidate: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Get fwd and bwd latency for layer.

    Parameters
    ----------
    config : MistConfig
        MistConfig object.
    layer_info : LayerInfo
        LayerInfo object.
    candidate : Tuple[int, int]
        (batch_size, num_gpus) candidate.
    """
    batch_size_per_device, tp_size = candidate

    factory_kwargs = {
        "gpu_gpu_bandwidth": config.gpu_gpu_bandwidth,
        "gpu_cpu_bandwidth": config.gpu_cpu_bandwidth,
    }
    fwd_strategy = layer_info.strategy.fwd_strategy
    symbol_mapping = {
        fwd_strategy.per_device_batch_size: batch_size_per_device,
        fwd_strategy.tp_size: tp_size,
    }
    fwd_latencies = 0
    bwd_latencies = 0
    for symbolic_node_spec in layer_info.fwd_info._symbolic_node_specs:
        concrete_node_spec = symbolic_node_spec.concretize(symbol_mapping)
        (fwd_latency, _, _), (bwd_latency, _, _) = concrete_node_spec.profile(
            **factory_kwargs
        )
        fwd_latencies += fwd_latency
        bwd_latencies += bwd_latency

    return fwd_latencies, bwd_latencies


def batched_latency_per_micro_batch(
    config: MistConfig,
    block_layer_info: LayerInfo,
    num_layers: int,
    pre_layer_info: Optional[LayerInfo],
    post_layer_info: Optional[LayerInfo],
):
    fwd_latency_func, bwd_latency_func = block_layer_info.get_latency_func(
        config=config,
    )


def batched_analyze_latencies(
    config: MistConfig,
    block_layer_info: LayerInfo,
    pre_layer_info: LayerInfo,
    post_layer_info: LayerInfo,
    block_layer_partition: List[int],
    num_micro_batches: int,
    candidates: Sequence[Any],
):
    num_stages = len(block_layer_partition)

    fwd_latencies = np.zeros((len(candidates), num_stages))
    bwd_latencies = np.zeros((len(candidates), num_stages))

    for stage_idx in range(num_stages):
        is_first_stage = stage_idx == 0
        is_last_stage = stage_idx == num_stages - 1
        candidates_for_stage = ...
        cur_fwd_latency, cur_bwd_latency = batched_latency_per_micro_batch(
            config=config,
            block_layer_info=block_layer_info,
            num_layers=block_layer_partition[stage_idx],
            pre_layer_info=pre_layer_info if is_first_stage else None,
            post_layer_info=post_layer_info if is_last_stage else None,
        )
        fwd_latencies[:stage_idx] = cur_fwd_latency
        bwd_latencies[:stage_idx] = cur_bwd_latency

    # pipe_fwd_latencies: [num_candidates]
    # pipe_bwd_latencies: [num_candidates]
    pipe_fwd_latencies = batched_latency_for_pipeline(
        num_stages=num_stages,
        num_micro_batches=num_micro_batches,
        latencies=fwd_latencies,
    )
    pipe_bwd_latencies = batched_latency_for_pipeline(
        num_stages=num_stages,
        num_micro_batches=num_micro_batches,
        latencies=bwd_latencies,
    )


def batched_latency_per_micro_batch_without_pre_post(
    config: MistConfig,
    layer_info: LayerInfo,
    num_layers: np.array,
    num_ckpts: np.array,
    fwd_parallelism: np.array,
    fwd_redundancy_elimination: np.array,
    fwd_offloading: np.array,
    bwd_parallelism: np.array,
    bwd_redundancy_elimination: np.array,
    bwd_offloading: np.array,
):
    pass


def batched_latency_per_layer(
    layer_info: LayerInfo,
):
    pass


class ModelGranularityOptimProb(OptimProb):
    strategy_granularity = "model"

    @property
    def combined_search_space(self):
        if not hasattr(self, "_combined_search_space"):
            combined_search_space = {}
            combined_search_space["num_ckpt_layers"] = list(
                range(max(self.block_layer_partition) + 1)
            )

            # Add layer strategy info for each layer
            for layer_name, sub_search_space in self.search_space.items():
                if sub_search_space is not None:
                    combined_search_space.update(
                        {
                            f"{layer_name}_{name}": candidates
                            for name, candidates in sub_search_space.items()
                        }
                    )

            combined_search_space_size = _calculate_search_space_size(
                *combined_search_space.values()
            )
            logger.info(f"Combined search space size: {combined_search_space_size}")

            self._combined_search_space = combined_search_space
            self._combined_search_space_size = combined_search_space_size

        return self._combined_search_space

    @property
    def combined_search_space_size(self):
        if not hasattr(self, "_combined_search_space_size"):
            self.combined_search_space
        return self._combined_search_space_size

    def randomly_create_individual(self):
        individual = [random.choice(self.combined_search_space.values())]
        return individual

    def sequentially_create_individual(self) -> Iterator[ModelStrategy]:
        return itertools.product(*self.combined_search_space.values())

    def batched_individuals(self, iterator, batch_size):
        return itertools.islice(iterator, batch_size)

    def make_numpy_individual(self, individual, batch_size):
        individual = tree_flatten(individual)[0]
        individual_np = np.array(individual, dtype=float)
        individual_np = individual_np.reshape((batch_size, -1))
        return individual_np

    def convert_individual_to_strategy_group(self, individual):
        ret = {}

        # Ckpt info
        ret["ckpts"] = individual[0]

        # Layer strategy info
        idx = 1
        for layer_name, sub_search_space in self.search_space.items():
            if layer_name == "pre_layer":
                share_strategy_for_fwd_bwd = (
                    self.config.pre_layer_share_strategy_for_fwd_bwd
                )
            elif layer_name == "post_layer":
                share_strategy_for_fwd_bwd = (
                    self.config.post_layer_share_strategy_for_fwd_bwd
                )
            elif layer_name == "block_layer":
                share_strategy_for_fwd_bwd = self.config.share_strategy_for_fwd_bwd
            else:
                raise ValueError(f"Unknown layer: {layer_name}")

            if sub_search_space is not None:
                layer_strategy_kwargs = {
                    name: individual[idx + i] for i, name in enumerate(sub_search_space)
                }
                # Post-processing the search space

                ret[layer_name] = LayerStrategy.from_search_space_sample(
                    share_strategy_for_fwd_bwd=share_strategy_for_fwd_bwd,
                    num_nodes=self.num_nodes_per_stage,
                    num_gpus_per_node=self.num_gpus_per_node_per_stage,
                    **layer_strategy_kwargs,
                )
                idx += len(sub_search_space)
            else:
                ret[layer_name] = None

        return ret

    def analyze_peak_memories(self, strategy_group):
        block_layer_strategy = strategy_group["block_layer"]
        pre_layer_strategy = strategy_group.get("pre_layer", None)
        post_layer_strategy = strategy_group.get("post_layer", None)

        peak_memories = []
        for stage_idx in range(self.num_stages):
            is_first_stage = stage_idx == 0
            is_last_stage = stage_idx == self.num_stages - 1
            num_ckpt_layers = min(
                strategy_group["ckpts"],
                self.block_layer_partition[stage_idx],
            )
            fwd_peak, bwd_peak, saved = analyze_module_memories_concrete_with_pre_post(
                block_layer_info=self.block_layer_info,
                block_layer_strategy=block_layer_strategy,
                num_ckpt_layers=num_ckpt_layers,
                num_layers=self.block_layer_partition[stage_idx],
                states_toggling=True,
                pre_layer_info=self.pre_layer_info if is_first_stage else None,
                pre_layer_strategy=pre_layer_strategy if is_first_stage else None,
                post_layer_info=self.post_layer_info if is_last_stage else None,
                post_layer_strategy=post_layer_strategy if is_last_stage else None,
            )

            warmup, _ = calculate_num_warmup_and_1f1b_phases(
                stage_idx, self.num_stages, self.gradient_accumulation_steps
            )

            fwd_peak_in_pipe_stage = warmup * saved + fwd_peak
            bwd_peak_in_pipe_stage = warmup * saved + bwd_peak
            peak_memories += [fwd_peak_in_pipe_stage, bwd_peak_in_pipe_stage]

        return peak_memories

    def analyze_latency(self, strategy_group):
        block_layer_strategy = strategy_group["block_layer"]
        pre_layer_strategy = strategy_group.get("pre_layer", None)
        post_layer_strategy = strategy_group.get("post_layer", None)

        # Construct the search space for block layer parallelism
        if not hasattr(self, "block_layer_parallelism_search_space"):
            self.block_layer_parallelism_search_space = tuple(
                set(self.block_layer_search_space["fwd_parallelism"])
            )

        # Analyze the latencies for each stage
        latencies = [[] for _ in range(self.num_stages)]
        fwd_latencies = []
        bwd_latencies = []
        for stage_idx in range(self.num_stages):
            is_first_stage = stage_idx == 0
            is_last_stage = stage_idx == self.num_stages - 1
            num_ckpt_layers = min(
                strategy_group["ckpts"],
                self.block_layer_partition[stage_idx],
            )
            fwd_latency, bwd_latency = analyze_module_latencies_concrete_with_pre_post(
                config=self.config,
                block_layer_info=self.block_layer_info,
                block_layer_strategy=block_layer_strategy,
                num_ckpt_layers=num_ckpt_layers,
                num_layers=self.block_layer_partition[stage_idx],
                block_layer_parallelism_search_space=self.block_layer_parallelism_search_space,
                pre_layer_info=self.pre_layer_info if is_first_stage else None,
                pre_layer_strategy=pre_layer_strategy if is_first_stage else None,
                post_layer_info=self.post_layer_info if is_last_stage else None,
                post_layer_strategy=post_layer_strategy if is_last_stage else None,
            )

            # There is no convertion overhead between different fwd-fwd and bwd-bwd phases
            # but there does exist convertion overhead between fwd-bwd phases, which is the 1f1b phase
            # TODO(zhanda): implement the following function
            # fwd_bwd_convertion_latency = calculate_convertion_latency(
            #     block_layer_strategy
            # )
            # fwd_latency += fwd_bwd_convertion_latency

            # Construct the latency sequence for the current stage
            warmup, fb = calculate_num_warmup_and_1f1b_phases(
                stage_idx, self.num_stages, self.gradient_accumulation_steps
            )
            for _ in range(warmup):
                latencies[stage_idx].append(fwd_latency)
            for _ in range(fb):
                latencies[stage_idx].append(fwd_latency)
                latencies[stage_idx].append(bwd_latency)
            for _ in range(warmup):
                latencies[stage_idx].append(bwd_latency)

            fwd_latencies.append(fwd_latency)
            bwd_latencies.append(bwd_latency)

        # latency = latency_for_pipe(
        #     num_stages=self.num_stages,
        #     num_micro_batches=self.gradient_accumulation_steps,
        #     latencies=latencies,
        # )
        pipe_fwd_latency = latency_for_pipe_with_fixed_time_in_stage(
            num_stages=self.num_stages,
            num_micro_batches=self.gradient_accumulation_steps,
            latencies=fwd_latencies,
        )
        pipe_bwd_latency = latency_for_pipe_with_fixed_time_in_stage(
            num_stages=self.num_stages,
            num_micro_batches=self.gradient_accumulation_steps,
            latencies=bwd_latencies,
        )
        latency = pipe_fwd_latency + pipe_bwd_latency

        return latency

    def evaluate(self, individual):
        time_starting = perf_counter()
        strategy_group = self.convert_individual_to_strategy_group(individual)
        time_strategy_group_converted = perf_counter()
        peak_memories = self.analyze_peak_memories(strategy_group)
        # logger.debug(f"Peak memories: {peak_memories}")
        penalties = sum(
            max(0, (peak_memory / GB - self.config.memory_capacity) * 100)
            for peak_memory in peak_memories
        )
        time_memories_analyzed = perf_counter()
        latency = self.analyze_latency(strategy_group)
        # logger.debug(f"Latency: {latency}")
        time_latency_analyzed = perf_counter()
        # logger.debug(
        #     f"Strategy group converted: {time_strategy_group_converted - time_starting:.6f}s, "
        #     f"peak memories analyzed: {time_memories_analyzed - time_strategy_group_converted:.6f}s, "
        #     f"latency analyzed: {time_latency_analyzed - time_memories_analyzed:.6f}s"
        # )
        return latency + penalties

    # Main entry point
    def tune(self):
        iterator, total_size = self.sequentially_create_individual()
        best_individual, best_objective = None, float("inf")
        for i, individual in tqdm(
            enumerate(iterator), disable=not self.tqdm_enabled, total=total_size
        ):
            objective = self.evaluate(individual)
            if objective < best_objective:
                logger.debug(f"Iter {i}: {individual} -> {objective}")
                best_individual, best_objective = individual, objective

            if i > 10:
                break

        return self.convert_individual_to_strategy_group(best_individual), objective

    def _debug(self):
        batch_size = 2
        iterator = chunked(self.sequentially_create_individual(), batch_size)
        for i, individuals in enumerate(iterator):
            individuals = self.make_numpy_individual(individuals, batch_size=batch_size)
            objectives = self.evaluate(individuals)

        return


class BatchAnalyzer:
    def __init__(
        self,
        block_layer_info: LayerInfo,
        pre_layer_info: LayerInfo,
        post_layer_info: LayerInfo,
    ):
        self.block_layer_info = block_layer_info
        self.pre_layer_info = pre_layer_info
        self.post_layer_info = post_layer_info

    def batch_analyze_latency(self, individuals):
        """
        batch_analyze_module_latencies_with_pre_post
        """
        pass

    def batch_analyze_module_latencies_with_pre_post(self, block_individuals):
        """
        batch_analyze_module_latencies_without_pre_post
        batch_analyze_layer_latencies
        """
        pass

    def batch_analyze_module_latencies_without_pre_post(self, individuals):
        """
        batch_analyze_layer_latencies
        """
        pass

    def batch_analyze_layer_latencies(self, layer_individuals):
        """
        analyze_layer_latencies
        """
        pass

    def analyze_layer_latencies(self, layer_individuals):
        """
        analyze_layer_exec_latency
        analyze_layer_comm_latency
        """
        pass

    def analyze_layer_exec_latency(self, individual, layer_type="block"):
        pass

    def analyze_layer_comm_latency(self, individual, layer_type="block"):
        pass

    def analyze_peak_memories(self, individuals):
        pass
