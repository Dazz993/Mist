import itertools
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, cache
from time import perf_counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional, Iterator, Any


import numpy as np
import sympy as sp
import torch

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
from mist.tuning.batched_model_optim_prob import batched_exec_latency_per_layer

logger = get_logger(__name__)

GB = 1024**3


@cache
def concretize_latency_info(
    config: MistConfig,
    layer_info,
    batch_size_per_device,
    tp_size,
):
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


def concrete_latency_info_with_different_fwd_bwd_strategy(
    config: MistConfig,
    layer_info,
    fwd_batch_size_per_device,
    fwd_tp_size,
    bwd_batch_size_per_device,
    bwd_tp_size,
):
    fwd_latencies, _ = concretize_latency_info(
        config, layer_info, fwd_batch_size_per_device, fwd_tp_size
    )
    _, bwd_latencies = concretize_latency_info(
        config, layer_info, bwd_batch_size_per_device, bwd_tp_size
    )
    return fwd_latencies, bwd_latencies


@cache
def analyze_layer_exec_latencies_symbolic(
    config: MistConfig,
    block_layer_info: LayerInfo,
    block_layer_parallelism_search_space: Tuple[Tuple[int, int, int, int, int]],
):
    bsz_symbol = sp.Symbol("_batch_size", integer=True, positive=True)
    tp_symbol = sp.Symbol("_tp_size", integer=True, positive=True)

    batch_size_and_tp_size_tuple_set = set(
        (batch_size, tp_size)
        for batch_size, dp_size, tp_size, ws_size, gs_size, oo_size in block_layer_parallelism_search_space
    )

    fwd_latencies = [(1e12, True)]
    bwd_latencies = [(1e12, True)]
    for batch_size_and_tp_size_tuple in batch_size_and_tp_size_tuple_set:
        batch_size, tp_size = batch_size_and_tp_size_tuple
        condition = sp.And(sp.Eq(bsz_symbol, batch_size), sp.Eq(tp_symbol, tp_size))
        fwd_latency, bwd_latency = concretize_latency_info(
            config, block_layer_info, batch_size, tp_size
        )
        fwd_latencies.append((fwd_latency, condition))
        bwd_latencies.append((bwd_latency, condition))
    fwd_latencies.append((1e12, True))
    bwd_latencies.append((1e12, True))

    fwd_latencies = sp.Piecewise(*fwd_latencies)
    bwd_latencies = sp.Piecewise(*bwd_latencies)
    symbol_mapping = {
        "batch_size_per_device": bsz_symbol,
        "tp_size": tp_symbol,
    }
    symbol_mapping.update(block_layer_info.strategy.generate_name_to_symbol_dict())
    fwd_latencies_func = autowrap_with_cython(
        tuple(symbol_mapping.values()), fwd_latencies
    )
    bwd_latencies_func = autowrap_with_cython(
        tuple(symbol_mapping.values()), bwd_latencies
    )
    return fwd_latencies_func, bwd_latencies_func, symbol_mapping


def analyze_module_latencies_concrete(
    config: MistConfig,
    block_layer_info: LayerInfo,
    block_layer_strategy: LayerStrategy,
    num_ckpt_layers: int,
    num_layers: int,
    block_layer_parallelism_search_space: Tuple[Tuple[int, int, int, int, int]],
):
    # Get the execution latencies
    (
        fwd_latencies_func,
        bwd_latencies_func,
        symbol_mapping,
    ) = analyze_layer_exec_latencies_symbolic(
        config, block_layer_info, block_layer_parallelism_search_space
    )

    # * Construct the symbol mapping
    symbol_mapping = {
        "batch_size_per_device": None,
        "tp_size": None,
    }
    symbol_mapping.update(block_layer_strategy.generate_name_to_symbol_dict())

    # * Evaluate the latencies for block layers
    symbol_mapping[
        "batch_size_per_device"
    ] = block_layer_strategy.fwd_strategy.per_device_batch_size
    symbol_mapping["tp_size"] = block_layer_strategy.fwd_strategy.tp_size
    fwd_latency = fwd_latencies_func(*symbol_mapping.values())

    symbol_mapping[
        "batch_size_per_device"
    ] = block_layer_strategy.bwd_strategy.per_device_batch_size
    symbol_mapping["tp_size"] = block_layer_strategy.bwd_strategy.tp_size
    bwd_latency = bwd_latencies_func(*symbol_mapping.values())

    fwd_latency = fwd_latency * num_layers
    bwd_latency = bwd_latency * num_layers + fwd_latency * num_ckpt_layers

    return fwd_latency, bwd_latency


def analyze_module_latencies_concrete_with_pre_post(
    config: MistConfig,
    block_layer_info: LayerInfo,
    block_layer_strategy: LayerStrategy,
    num_ckpt_layers: int,
    num_layers: int,
    block_layer_parallelism_search_space: Tuple[Tuple[int, int, int, int, int]],
    pre_layer_info: Optional[LayerInfo] = None,
    pre_layer_strategy: Optional[LayerStrategy] = None,
    post_layer_info: Optional[LayerInfo] = None,
    post_layer_strategy: Optional[LayerStrategy] = None,
):
    fwd_latency, bwd_latency = analyze_module_latencies_concrete(
        config,
        block_layer_info,
        block_layer_strategy,
        num_ckpt_layers,
        num_layers,
        block_layer_parallelism_search_space,
    )
    logger.debug(f"fwd_latency: {fwd_latency}, bwd_latency: {bwd_latency}")

    # If pre- and post- layers are not None, we need to add their latencies
    if pre_layer_info is not None:
        assert pre_layer_strategy is not None
        (
            fwd_latency_pre,
            bwd_latency_pre,
        ) = concrete_latency_info_with_different_fwd_bwd_strategy(
            config,
            pre_layer_info,
            pre_layer_strategy.fwd_strategy.per_device_batch_size,
            pre_layer_strategy.fwd_strategy.tp_size,
            pre_layer_strategy.bwd_strategy.per_device_batch_size,
            pre_layer_strategy.bwd_strategy.tp_size,
        )
        fwd_latency += fwd_latency_pre
        bwd_latency += bwd_latency_pre

        # logger.debug(
        #     f"fwd_latency_pre: {fwd_latency_pre}, bwd_latency_pre: {bwd_latency_pre}"
        # )

    if post_layer_info is not None:
        assert post_layer_strategy is not None
        (
            fwd_latency_post,
            bwd_latency_post,
        ) = concrete_latency_info_with_different_fwd_bwd_strategy(
            config,
            post_layer_info,
            post_layer_strategy.fwd_strategy.per_device_batch_size,
            post_layer_strategy.fwd_strategy.tp_size,
            post_layer_strategy.bwd_strategy.per_device_batch_size,
            post_layer_strategy.bwd_strategy.tp_size,
        )
        fwd_latency += fwd_latency_post
        bwd_latency += bwd_latency_post

        # logger.debug(
        #     f"fwd_latency_post: {fwd_latency_post}, bwd_latency_post: {bwd_latency_post}"
        # )

    return fwd_latency, bwd_latency


# def calculate_convertion_latency(
#     layer_info: LayerInfo,
#     concrete_strategy: LayerStrategy,
# ):
#     """Calculate the state convertion latency between two strategies

#     1. convertion latency of tp
#     2. convertion latency of different offloadings
#     """
#     pre_tp_size = layer_info.strategy.tp_size


def concretize_memory_info(
    layer_info: LayerInfo, concrete_strategy: LayerStrategy, ckpt=None
):
    fwd_name_to_symbol = concrete_strategy.fwd_strategy.generate_name_to_symbol_dict()
    bwd_name_to_symbol = concrete_strategy.bwd_strategy.generate_name_to_symbol_dict()
    if fwd_name_to_symbol["ckpt"] is None and ckpt is not None:
        fwd_name_to_symbol["ckpt"] = ckpt
    if bwd_name_to_symbol["ckpt"] is None and ckpt is not None:
        bwd_name_to_symbol["ckpt"] = ckpt

    # symbol_mapping = layer_info.strategy.generate_symbol_mapping(concrete_strategy)
    # ori_ckpt_symbol = getattr(layer_info.strategy, "ckpt")
    # if ori_ckpt_symbol is None and ckpt is not None:
    #     symbol_mapping[ori_ckpt_symbol] = ckpt

    fwd_info = layer_info.fwd_info
    bwd_info = layer_info.bwd_info

    # # Common
    full_weights = fwd_info.get_concrete_memory("full_weights", fwd_name_to_symbol)
    full_grads = fwd_info.get_concrete_memory("full_grads", fwd_name_to_symbol)
    full_opts = fwd_info.get_concrete_memory("full_opts", fwd_name_to_symbol)
    fwd_partial_weights = fwd_info.get_concrete_memory(
        "partial_weights", fwd_name_to_symbol
    )
    fwd_partial_grads = fwd_info.get_concrete_memory(
        "partial_grads", fwd_name_to_symbol
    )
    fwd_partial_opts = fwd_info.get_concrete_memory("partial_opts", fwd_name_to_symbol)
    bwd_partial_weights = bwd_info.get_concrete_memory(
        "partial_weights", bwd_name_to_symbol
    )
    bwd_partial_grads = bwd_info.get_concrete_memory(
        "partial_grads", bwd_name_to_symbol
    )
    bwd_partial_opts = bwd_info.get_concrete_memory("partial_opts", bwd_name_to_symbol)
    # Ckpt
    saved = fwd_info.get_concrete_memory("saved", fwd_name_to_symbol)
    fwd_peak = fwd_info.get_concrete_memory("peak", fwd_name_to_symbol)
    bwd_peak = bwd_info.get_concrete_memory("peak", bwd_name_to_symbol)
    # fwd_partial_weights = gsm.subs(fwd_info.partial_weights, symbol_mapping)
    # fwd_partial_grads = gsm.subs(fwd_info.partial_grads, symbol_mapping)
    # fwd_partial_opts = gsm.subs(fwd_info.partial_opts, symbol_mapping)
    # bwd_partial_weights = gsm.subs(bwd_info.partial_weights, symbol_mapping)
    # bwd_partial_grads = gsm.subs(bwd_info.partial_grads, symbol_mapping)
    # bwd_partial_opts = gsm.subs(bwd_info.partial_opts, symbol_mapping)
    # # Ckpt
    # saved = gsm.subs(fwd_info.saved, symbol_mapping)
    # fwd_peak = gsm.subs(fwd_info.peak, symbol_mapping)
    # bwd_peak = gsm.subs(bwd_info.peak, symbol_mapping)

    return {
        "full_weights": full_weights,
        "full_grads": full_grads,
        "full_opts": full_opts,
        "fwd_partial_weights": fwd_partial_weights,
        "fwd_partial_grads": fwd_partial_grads,
        "fwd_partial_opts": fwd_partial_opts,
        "bwd_partial_weights": bwd_partial_weights,
        "bwd_partial_grads": bwd_partial_grads,
        "bwd_partial_opts": bwd_partial_opts,
        "saved": saved,
        "fwd_peak": fwd_peak,
        "bwd_peak": bwd_peak,
    }


@cache
def analyze_module_memories_symbolic(
    block_layer_info: LayerInfo,
    num_layers: int,
    states_toggling: bool,
):
    """Symbolic analyze the peak memory of a module
    The memory can be calculated by the following formula:
    peak_memory = saved_memory_prev + peak_memory_curr + state_memory_curr
    saved_memory_curr = saved_memory_before + saved_memory_curr

    Notes:
    1. The assumption is all layers are using the same strategy
    2. Pre- and post- layers should be considered separately outside this function
    """
    num_ckpt_layers = sp.Symbol("_num_ckpt_layers", integer=True, nonnegative=True)

    def _subs(layer_info: LayerInfo, field: str, ckpt: Union[sp.Symbol, int]):
        return getattr(layer_info, field).subs({layer_info.strategy.ckpt: ckpt})

    fwd_info = block_layer_info.fwd_info
    bwd_info = block_layer_info.bwd_info

    # Common
    full_weights_per_layer = fwd_info.full_weights
    full_grads_per_layer = fwd_info.full_grads
    full_opts_per_layer = fwd_info.full_opts
    fwd_partial_weights_per_layer = fwd_info.partial_weights
    fwd_partial_grads_per_layer = fwd_info.partial_grads
    fwd_partial_opts_per_layer = fwd_info.partial_opts
    bwd_partial_weights_per_layer = bwd_info.partial_weights
    bwd_partial_grads_per_layer = bwd_info.partial_grads
    bwd_partial_opts_per_layer = bwd_info.partial_opts
    # Composition
    fw_fpg_fpo = (
        full_weights_per_layer
        + fwd_partial_grads_per_layer
        + fwd_partial_opts_per_layer
    )
    fw_fg_bpo = (
        full_weights_per_layer + full_grads_per_layer + fwd_partial_opts_per_layer
    )
    fpw_fpg_fpo = (
        fwd_partial_weights_per_layer
        + fwd_partial_grads_per_layer
        + fwd_partial_opts_per_layer
    )
    bpw_bpg_bpo = (
        bwd_partial_weights_per_layer
        + bwd_partial_grads_per_layer
        + bwd_partial_opts_per_layer
    )
    # Ckpt-related
    saved_per_layer = _subs(fwd_info, "saved", ckpt=0)
    saved_ckpt_per_layer = _subs(fwd_info, "saved", ckpt=1)
    fwd_peak_per_layer = _subs(fwd_info, "peak", ckpt=0)
    fwd_peak_ckpt_per_layer = _subs(fwd_info, "peak", ckpt=1)
    bwd_peak_per_layer = _subs(bwd_info, "peak", ckpt=0)
    bwd_peak_ckpt_per_layer = _subs(bwd_info, "peak", ckpt=1)

    # First num_ckpt_layers layer are ckpt layers
    # consider the last layer of the first num_ckpt_layers layers
    all_ckpt = sp.Eq(num_ckpt_layers, num_layers)
    no_ckpt = sp.Eq(num_ckpt_layers, 0)
    if states_toggling:
        fwd_states = fw_fpg_fpo * 2 + bpw_bpg_bpo * (num_layers - 2)
        bwd_states = fw_fg_bpo * 2 + bpw_bpg_bpo * (num_layers - 2)
    else:
        fwd_states = fw_fpg_fpo * 2 + fpw_fpg_fpo * (num_layers - 2)
        bwd_states = fw_fg_bpo * 2 + fpw_fpg_fpo * (num_layers - 2)
    # case 1: suppose no ckpt
    saved_no_ckpt = saved_per_layer * num_layers
    curr_saved_no_ckpt = saved_per_layer * (num_layers - 1)
    fwd_peak_no_ckpt = curr_saved_no_ckpt + fwd_states + fwd_peak_per_layer
    bwd_peak_no_ckpt = curr_saved_no_ckpt + bwd_states + bwd_peak_per_layer
    # case 2: suppose all ckpt
    saved_all_ckpt = saved_ckpt_per_layer * num_layers
    curr_saved_all_ckpt = saved_ckpt_per_layer * (num_layers - 1)
    fwd_peak_ckpt = curr_saved_all_ckpt + fwd_states + fwd_peak_ckpt_per_layer
    bwd_peak_ckpt = curr_saved_all_ckpt + bwd_states + bwd_peak_ckpt_per_layer
    # case 3: suppose some ckpt
    # we assume the peak is the last layer
    saved_some_ckpt = saved_ckpt_per_layer * num_ckpt_layers + saved_per_layer * (
        num_layers - num_ckpt_layers
    )
    curr_saved_some_ckpt = saved_ckpt_per_layer * (
        num_ckpt_layers - 1
    ) + saved_per_layer * (num_layers - num_ckpt_layers)
    fwd_peak_some_ckpt = curr_saved_some_ckpt + fwd_states + fwd_peak_ckpt_per_layer
    bwd_peak_some_ckpt = curr_saved_some_ckpt + bwd_states + bwd_peak_ckpt_per_layer

    # Summary
    fwd_peak = sp.Piecewise(
        (fwd_peak_no_ckpt, no_ckpt),
        (fwd_peak_ckpt, all_ckpt),
        (fwd_peak_some_ckpt, True),
    )
    bwd_peak = sp.Piecewise(
        (bwd_peak_no_ckpt, no_ckpt),
        (bwd_peak_ckpt, all_ckpt),
        (bwd_peak_some_ckpt, True),
    )
    saved = sp.Piecewise(
        (saved_no_ckpt, no_ckpt),
        (saved_all_ckpt, all_ckpt),
        (saved_some_ckpt, True),
    )
    symbol_mapping = {"num_ckpt_layers": num_ckpt_layers}
    symbol_mapping.update(block_layer_info.strategy.generate_name_to_symbol_dict())

    fwd_peak_func = autowrap_with_cython(tuple(symbol_mapping.values()), fwd_peak)
    bwd_peak_func = autowrap_with_cython(tuple(symbol_mapping.values()), bwd_peak)
    saved_func = autowrap_with_cython(tuple(symbol_mapping.values()), saved)
    return fwd_peak_func, bwd_peak_func, saved_func, symbol_mapping

    # return fwd_peak, bwd_peak, saved, symbol_mapping


def analyze_module_memories_concrete_with_pre_post(
    block_layer_info: LayerInfo,
    block_layer_strategy: LayerStrategy,
    num_ckpt_layers: int,
    num_layers: int,
    states_toggling: bool,
    pre_layer_info: Optional[LayerInfo] = None,
    pre_layer_strategy: Optional[LayerStrategy] = None,
    post_layer_info: Optional[LayerInfo] = None,
    post_layer_strategy: Optional[LayerStrategy] = None,
):
    (
        fwd_peak_func,
        bwd_peak_func,
        saved_func,
        symbol_mapping,
    ) = analyze_module_memories_symbolic(block_layer_info, num_layers, states_toggling)

    # Construct the symbol mapping
    symbol_mapping = {"num_ckpt_layers": num_ckpt_layers}
    symbol_mapping.update(block_layer_strategy.generate_name_to_symbol_dict())

    # Evaluate the peak memory for block layers
    fwd_peak = fwd_peak_func(*symbol_mapping.values())
    bwd_peak = bwd_peak_func(*symbol_mapping.values())
    saved = saved_func(*symbol_mapping.values())

    # Deal with pre- and post- layers
    if pre_layer_info is not None:
        assert pre_layer_strategy is not None
        concrete_pre_layer_info = concretize_memory_info(
            pre_layer_info, pre_layer_strategy
        )
        saved += concrete_pre_layer_info["saved"]
        fwd_peak += concrete_pre_layer_info["saved"]
        bwd_peak += concrete_pre_layer_info["saved"]
        if states_toggling:
            fwd_peak += concrete_pre_layer_info["bwd_partial_weights"]
            fwd_peak += concrete_pre_layer_info["bwd_partial_grads"]
            fwd_peak += concrete_pre_layer_info["bwd_partial_opts"]
            bwd_peak += concrete_pre_layer_info["fwd_partial_weights"]
            bwd_peak += concrete_pre_layer_info["fwd_partial_grads"]
            bwd_peak += concrete_pre_layer_info["fwd_partial_opts"]
        else:
            fwd_peak += concrete_pre_layer_info["fwd_partial_weights"]
            fwd_peak += concrete_pre_layer_info["fwd_partial_grads"]
            fwd_peak += concrete_pre_layer_info["fwd_partial_opts"]
            bwd_peak += concrete_pre_layer_info["bwd_partial_weights"]
            bwd_peak += concrete_pre_layer_info["bwd_partial_grads"]
            bwd_peak += concrete_pre_layer_info["bwd_partial_opts"]

    if post_layer_info is not None:
        assert post_layer_strategy is not None
        concrete_post_layer_info = concretize_memory_info(
            post_layer_info, post_layer_strategy
        )
        saved += concrete_post_layer_info["saved"]
        # TODO(zhanda): Should have a better estimation of the peak memory when including post-layer.
        # The current implementation is a very rough estimation.
        fwd_peak += concrete_post_layer_info["fwd_peak"]
        bwd_peak += concrete_post_layer_info["bwd_peak"]
        # States
        fwd_peak += concrete_post_layer_info["full_weights"]
        fwd_peak += concrete_post_layer_info["fwd_partial_grads"]
        fwd_peak += concrete_post_layer_info["fwd_partial_opts"]
        bwd_peak += concrete_post_layer_info["full_weights"]
        bwd_peak += concrete_post_layer_info["full_grads"]
        bwd_peak += concrete_post_layer_info["bwd_partial_opts"]

    return fwd_peak, bwd_peak, saved


class ModelGranularityOptimProb(OptimProb):
    strategy_granularity = "model"

    def randomly_create_individual(self):
        individual = []

        # Add ckpt info for each stage
        for stage_idx in range(self.num_stages):
            num_layers_in_stage = self.block_layer_partition[stage_idx]
            individual.append(random.choice(list(range(num_layers_in_stage + 1))))

        # Add layer strategy info for each layer
        for layer_name, sub_search_space in self.search_space.items():
            if sub_search_space is not None:
                individual += [
                    random.choice(candidates)
                    for name, candidates in sub_search_space.items()
                ]
        return individual

    def sequentially_create_individual(self) -> Iterator[ModelStrategy]:
        combined_search_space = {}

        # Add ckpt info for each stage
        # for stage_idx in range(self.num_stages):
        #     num_layers_in_stage = self.block_layer_partition[stage_idx]
        #     combined_search_space[f"stage_{stage_idx}_num_ckpt_layers"] = list(
        #         range(num_layers_in_stage + 1)
        #     )
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
        return (
            itertools.product(*combined_search_space.values()),
            combined_search_space_size,
        )

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
            logger.debug(fwd_latency, bwd_latency)

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
        return
        block_layer_parallelism_search_space = self.block_layer_search_space[
            "fwd_parallelism"
        ]

        # Latency
        def latency_preprocessing():
            (
                block_layer_exec_fwd_latencies,
                block_layer_exec_bwd_latencies,
            ) = batched_exec_latency_per_layer(
                config=self.config,
                layer_info=self.block_layer_info,
                candidates=candidates,
            )

            (
                block_layer_overlapped_comm_fwd_latencies,
                block_layer_overlapped_comm_bwd_latencies,
            ) = batched_overlapped_comm_latency_per_layer(
                config=self.config,
                layer_info=self.block_layer_info,
                candidates=candidates,
            )

        def analyze_latency(candidates):
            pass

        batch_size_and_tp_size_pair_set = set(
            (batch_size, tp_size)
            for batch_size, dp_size, tp_size, ws_size, gs_size, oo_size in block_layer_parallelism_search_space
        )
        (
            layer_exec_fwd_latencies,
            layer_exec_bwd_latencies,
        ) = batched_exec_latency_per_layer(
            config=self.config,
            layer_info=self.block_layer_info,
            candidates=batch_size_and_tp_size_pair_set,
        )

        # 2. Get layer overlapped comm latencies
