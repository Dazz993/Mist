import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, cache
from time import perf_counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional


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

POWER_OF_TWO = [2**i for i in range(15)]


def _calculate_search_space_size(*iterables):
    search_space_size = 1
    for candidates in iterables:
        search_space_size *= len(candidates)
    return search_space_size


class OptimProb:
    strategy_granularity: str

    def __init__(
        self,
        config: MistConfig,
        num_nodes_per_stage: int,
        num_gpus_per_node_per_stage: int,
        gradient_accumulation_steps: int,
        block_layer_info: LayerInfo,
        block_layer_partition: List[int] = None,
        pre_layer_info: Optional[LayerInfo] = None,
        post_layer_info: Optional[LayerInfo] = None,
        tqdm_enabled: bool = True,
    ):
        self.config = config
        self.num_nodes_per_stage = num_nodes_per_stage
        self.num_gpus_per_node_per_stage = num_gpus_per_node_per_stage
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.block_layer_info = block_layer_info
        self.block_layer_partition = block_layer_partition
        self.num_stages = len(block_layer_partition) if block_layer_partition else 1
        self.pre_layer_info = pre_layer_info
        self.post_layer_info = post_layer_info
        self.tqdm_enabled = tqdm_enabled

        # Create search space
        _create_search_space_for_a_layer = partial(
            create_search_space_for_a_layer,
            config=config,
            num_nodes=num_nodes_per_stage,
            num_gpus_per_node=num_gpus_per_node_per_stage,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        self.block_layer_search_space = _create_search_space_for_a_layer(
            layer_name="block_layer"
        )
        self.pre_layer_search_space = None
        self.post_layer_search_space = None
        if self.pre_layer_info is not None:
            self.pre_layer_search_space = _create_search_space_for_a_layer(
                layer_name="pre_layer"
            )
        if self.post_layer_info is not None:
            self.post_layer_search_space = _create_search_space_for_a_layer(
                layer_name="post_layer"
            )

        self.search_space = {
            "block_layer": self.block_layer_search_space,
            "pre_layer": self.pre_layer_search_space,
            "post_layer": self.post_layer_search_space,
        }

        # Run setup
        self.setup()

    def setup(self):
        pass

    def tune(self):
        raise NotImplementedError


def create_search_space_for_a_layer(
    layer_name: str,
    config: MistConfig,
    num_nodes: int,
    num_gpus_per_node: int,
    gradient_accumulation_steps: int,
):
    num_gpus = num_nodes * num_gpus_per_node
    batch_size_per_micro_batch = config.global_batch_size // gradient_accumulation_steps

    search_space = {}

    if layer_name.startswith("pre_layer"):
        ckpt_tuning_enabled = config.pre_layer_ckpt_tuning_enabled
        ckpt_default = 0
        offloading_enabled = config.pre_layer_offloading_enabled
        redundancy_sharding_enabled = config.pre_layer_redundancy_sharding_enabled
        share_strategy_for_fwd_bwd = config.pre_layer_share_strategy_for_fwd_bwd
    elif layer_name.startswith("post_layer"):
        ckpt_tuning_enabled = config.post_layer_ckpt_tuning_enabled
        ckpt_default = 0
        offloading_enabled = config.post_layer_offloading_enabled
        redundancy_sharding_enabled = config.post_layer_redundancy_sharding_enabled
        share_strategy_for_fwd_bwd = config.post_layer_share_strategy_for_fwd_bwd
    else:
        # ckpt_tuning_enabled = config.ckpt_tuning_enabled
        # ckpt_default = True
        offloading_enabled = config.offloading_enabled
        redundancy_sharding_enabled = config.redundancy_sharding_enabled
        share_strategy_for_fwd_bwd = config.share_strategy_for_fwd_bwd

    # CKPT
    # now only apply to pre_layer and post_layer
    if layer_name.startswith(("pre_layer", "post_layer")):
        if ckpt_tuning_enabled:
            ckpt_candidates = [0, 1]
        else:
            ckpt_candidates = [ckpt_default]
        search_space["ckpt"] = ckpt_candidates

    # Phase Strategy
    # * Parallelism
    parallelism_candidates = []
    for batch_size in config.batch_size_candidates:
        # DP Size is determined by local batch size and global batch size
        if (
            batch_size > batch_size_per_micro_batch
            or batch_size_per_micro_batch % batch_size != 0
        ):
            continue
        dp_size = batch_size_per_micro_batch // batch_size

        # TP size is determined by DP size and number of GPUs
        if dp_size > num_gpus or num_gpus % dp_size != 0:
            continue
        tp_size = num_gpus // dp_size

        redundancy_sharding_choices = [n for n in POWER_OF_TWO if n <= num_gpus]
        oo_size = redundancy_sharding_choices[-1]
        if redundancy_sharding_enabled:
            # Sharding size is bounded by the number of GPUs
            for gs_size in redundancy_sharding_choices:
                ws_size = 1
                parallelism_candidates.append(
                    (batch_size, dp_size, tp_size, ws_size, gs_size, oo_size)
                )
            for ws_size in redundancy_sharding_choices:
                gs_size = redundancy_sharding_choices[-1]
                parallelism_candidates.append(
                    (batch_size, dp_size, tp_size, ws_size, gs_size, oo_size)
                )
        else:
            ws_size = 1
            gs_size = 1
            parallelism_candidates.append(
                (batch_size, dp_size, tp_size, ws_size, gs_size, oo_size)
            )

    # * Offloading
    if offloading_enabled:
        offloading_ratio_choices = [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]
        offloading_ratio_candidates = []
        # for oo_ratio in offloading_ratio_choices:
        #     wo_ratio = 0
        #     go_ratio = 0
        #     offloading_ratio_candidates.append((wo_ratio, go_ratio, oo_ratio))
        for go_ratio in offloading_ratio_choices:
            wo_ratio = 0
            oo_ratio = offloading_ratio_choices[-1]
            offloading_ratio_candidates.append((wo_ratio, go_ratio, oo_ratio))
        for wo_ratio in offloading_ratio_choices:
            go_ratio = offloading_ratio_choices[-1]
            oo_ratio = offloading_ratio_choices[-1]
            offloading_ratio_candidates.append((wo_ratio, go_ratio, oo_ratio))
    else:
        offloading_ratio_candidates = [(0.0, 0.0, 0.0)]

    if share_strategy_for_fwd_bwd:
        search_space["fwd_parallelism"] = parallelism_candidates
        search_space["fwd_offloading"] = offloading_ratio_candidates
    else:
        search_space["fwd_parallelism"] = parallelism_candidates
        search_space["fwd_offloading"] = offloading_ratio_candidates
        search_space["bwd_parallelism"] = parallelism_candidates
        search_space["bwd_offloading"] = offloading_ratio_candidates

    return search_space
