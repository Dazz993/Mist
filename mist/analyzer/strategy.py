from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from numbers import Number
from typing import Callable, Sequence, List, Dict, Any, Optional, Union, Tuple
import functools
from functools import partial
from tqdm import tqdm

import numpy as np
import torch
import sympy as sp

from mist import global_symbol_manager as gsm
from mist.config import MistConfig
from mist.analyzer.recorder import ExecInfoRecorder, ExecType
from mist.analyzer.info import LayerInfo
from mist.utils.pipeline_parallel import calculate_num_warmup_and_1f1b_phases


Var = Union[sp.Basic, Number]
VarInt = Union[sp.Basic, int]
VarFloat = Union[sp.Basic, float]
VarBool = Union[sp.Basic, bool]

POWER_OF_TWO = [2**i for i in range(15)]


@dataclass
class ModelStrategy:
    strategy_granularity: str = "stage"
    strategies: List[List[List[LayerStrategy]]] = None
    gradient_accumulation_steps: int = 1
    block_layer_partition: List[List[int]] = None
    pre_layer_strategies: Optional[List[LayerStrategy]] = None
    post_layer_strategies: Optional[List[LayerStrategy]] = None

    def __post_init__(self):
        self.assert_is_valid()

    def assert_is_valid(self):
        num_stages = len(self.strategies) if self.strategies is not None else 1
        if self.strategy_granularity == "model":
            assert len(self.strategies) == 1  # only one stage strategy
            assert len(self.strategies[0]) == 1  # only one micro batch strategy
            assert len(self.strategies[0][0]) == 1  # only one layer strategy
        elif self.strategy_granularity == "stage":
            assert len(self.strategies) == num_stages  # one stage strategy per stage
            for stage_strategy in self.strategies:
                assert (
                    len(stage_strategy) == 1
                )  # only one micro batch strategy per stage
                assert len(stage_strategy[0]) == 1  # only one layer strategy per stage
        elif self.strategy_granularity == "micro_batch":
            assert len(self.strategies) == num_stages  # one stage strategy per stage
            for stage_idx, stage_strategy in enumerate(self.strategies):
                warmup, _ = calculate_num_warmup_and_1f1b_phases(
                    stage_idx, num_stages, self.gradient_accumulation_steps
                )
                assert len(stage_strategy) == (
                    warmup + 1
                )  # one micro batch strategy per warmup and for a 1f1b
                assert all(
                    len(mb_strategy) == 1 for mb_strategy in stage_strategy
                )  # only one layer strategy per micro batch
        else:
            raise ValueError(
                f"Unknown strategy granularity: {self.strategy_granularity}"
            )

    def get_block_strategy(self, stage_idx=0, mb_idx=0, layer_idx=0, mb_type=None):
        if self.strategy_granularity == "model":
            return self.strategies[0][0][0]
        elif self.strategy_granularity == "stage":
            return self.strategies[stage_idx][0][0]
        elif self.strategy_granularity == "micro_batch":
            if mb_type == "warmup":
                return self.strategies[stage_idx][mb_idx][0]
            elif mb_type == "1f1b":
                return self.strategies[stage_idx][-1][0]
            elif mb_type == "cooldown":
                return self.strategies[stage_idx][-2 - mb_idx][0]
        elif self.strategy_granularity == "layer":
            raise NotImplementedError("Layer granularity is not supported yet.")
        else:
            raise ValueError(
                f"Unknown strategy granularity: {self.strategy_granularity}"
            )

    def get_pre_layer_strategy(self, mb_idx=0):
        if self.strategy_granularity in {"model", "stage"}:
            return self.pre_layer_strategies[0]
        else:
            return self.pre_layer_strategies[mb_idx]

    def get_post_layer_strategy(self, mb_idx=0):
        if self.strategy_granularity in {"model", "stage"}:
            return self.post_layer_strategies[0]
        else:
            return self.post_layer_strategies[mb_idx]


def create_model_strategy_from_layer_strategies(
    block_layer_strategy_provider,
    config: MistConfig,
    block_layer_partition: List[int],
    gradient_accumulation_steps: int,
    pre_layer_strategy_provider=None,
    post_layer_strategy_provider=None,
):
    num_stages = len(block_layer_partition)
    # Strategies is a 3D list: [stage_idx][mb_idx][layer_idx]
    strategies = []
    # Pre- and post- layer strategies are 1D list: [mb_idx]
    pre_layer_strategies = []
    post_layer_strategies = []
    if config.strategy_granularity == "model":
        strategies = [[[block_layer_strategy_provider()]]]
        pre_layer_strategies = [pre_layer_strategy_provider()]
        post_layer_strategies = [post_layer_strategy_provider()]
    elif config.strategy_granularity == "stage":
        strategies = [[[block_layer_strategy_provider()]] for _ in range(num_stages)]
        pre_layer_strategies = [pre_layer_strategy_provider()]
        post_layer_strategies = [pre_layer_strategy_provider()]
    elif config.strategy_granularity == "micro_batch":
        for stage_idx in range(num_stages):
            warmup, _ = calculate_num_warmup_and_1f1b_phases(
                stage_idx, num_stages, gradient_accumulation_steps
            )
            stage_strategies = [
                [block_layer_strategy_provider()] for _ in range(warmup + 1)
            ]
            strategies.append(stage_strategies)
            pre_layer_strategies.append(pre_layer_strategy_provider())
            post_layer_strategies.append(post_layer_strategy_provider())
    elif config.strategy_granularity == "layer":
        raise NotImplementedError("Layer granularity is not supported yet.")
    else:
        raise ValueError(f"Unknown strategy granularity: {config.strategy_granularity}")

    return ModelStrategy(
        strategy_granularity=config.strategy_granularity,
        strategies=strategies,
        gradient_accumulation_steps=gradient_accumulation_steps,
        block_layer_partition=block_layer_partition,
        pre_layer_strategies=pre_layer_strategies,
        post_layer_strategies=post_layer_strategies,
    )


@dataclass
class LayerStrategy:
    num_nodes: VarInt = 1
    num_gpus_per_node: VarInt = 1
    ckpt: VarInt = 1
    share_strategy_for_fwd_bwd: bool = False
    fwd_strategy: PhaseStrategy = None
    bwd_strategy: PhaseStrategy = None

    @classmethod
    def from_search_space_sample(
        cls,
        share_strategy_for_fwd_bwd: bool,
        num_nodes: VarInt,
        num_gpus_per_node: VarInt,
        fwd_parallelism: Tuple[VarInt, VarInt, VarInt, VarInt, VarInt, VarInt],
        fwd_offloading: Tuple[VarFloat, VarFloat, VarFloat],
        bwd_parallelism: Optional[
            Tuple[VarInt, VarInt, VarInt, VarInt, VarInt, VarInt]
        ] = None,
        bwd_offloading: Optional[Tuple[VarFloat, VarFloat, VarFloat]] = None,
        ckpt: VarInt = None,
    ):
        fwd_strategy = PhaseStrategy.from_search_space_sample(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            ckpt=ckpt,
            parallelism=fwd_parallelism,
            offloading=fwd_offloading,
        )

        if share_strategy_for_fwd_bwd:
            bwd_strategy = fwd_strategy
        else:
            bwd_strategy = PhaseStrategy.from_search_space_sample(
                num_nodes=num_nodes,
                num_gpus_per_node=num_gpus_per_node,
                ckpt=ckpt,
                parallelism=bwd_parallelism,
                offloading=bwd_offloading,
            )

        return cls(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            ckpt=ckpt,
            share_strategy_for_fwd_bwd=share_strategy_for_fwd_bwd,
            fwd_strategy=fwd_strategy,
            bwd_strategy=bwd_strategy,
        )

    def generate_symbol_mapping(self, concrete_layer_strategy):
        concrete_fwd_strategy = concrete_layer_strategy.fwd_strategy
        concrete_bwd_strategy = concrete_layer_strategy.bwd_strategy
        mapping = {}
        mapping.update(self.fwd_strategy.generate_symbol_mapping(concrete_fwd_strategy))
        mapping.update(self.bwd_strategy.generate_symbol_mapping(concrete_bwd_strategy))
        return mapping

    def generate_name_to_symbol_dict(self):
        name_to_symbol_dict = {}
        fwd_name_to_symbol_dict = self.fwd_strategy.generate_name_to_symbol_dict()
        fwd_name_to_symbol_dict.pop("ckpt")
        bwd_name_to_symbol_dict = self.bwd_strategy.generate_name_to_symbol_dict()
        bwd_name_to_symbol_dict.pop("num_nodes")
        bwd_name_to_symbol_dict.pop("num_gpus_per_node")
        bwd_name_to_symbol_dict.pop("ckpt")
        fwd_name_to_symbol_dict = {
            f"fwd_{name}": symbol for name, symbol in fwd_name_to_symbol_dict.items()
        }
        bwd_name_to_symbol_dict = {
            f"bwd_{name}": symbol for name, symbol in bwd_name_to_symbol_dict.items()
        }
        name_to_symbol_dict.update(fwd_name_to_symbol_dict)
        name_to_symbol_dict.update(bwd_name_to_symbol_dict)

        # Remove the duplicated symbols

        return name_to_symbol_dict


@dataclass
class PhaseStrategy:
    # Parent layer strategy
    layer_strategy: Optional[LayerStrategy] = None
    # Inherited from the parent layer strategy
    num_nodes: VarInt = 1
    num_gpus_per_node: VarInt = 1
    ckpt: VarInt = 1

    # Training
    per_device_batch_size: VarInt = 1
    # Parallelism
    dp_size: VarInt = 1
    tp_size: VarInt = 1
    ws_size: VarInt = 1
    gs_size: VarInt = 1
    os_size: VarInt = 1

    # Memory optimization
    wo_ratio: VarFloat = 0.0
    go_ratio: VarFloat = 0.0
    oo_ratio: VarFloat = 0.0

    # Mapping
    mapping: Optional[Dict[sp.Basic, Var]] = None
    maybe_symbols: Tuple[str, ...] = (
        "num_nodes",
        "num_gpus_per_node",
        "ckpt",
        "per_device_batch_size",
        "dp_size",
        "tp_size",
        "ws_size",
        "gs_size",
        "os_size",
        "wo_ratio",
        "go_ratio",
        "oo_ratio",
    )

    def __post_init__(self):
        if self.layer_strategy is not None:
            self.num_nodes = self.layer_strategy.num_nodes
            self.num_gpus_per_node = self.layer_strategy.num_gpus_per_node
            self.ckpt = self.layer_strategy.ckpt

    def generate_symbol_mapping(self, concrete_phase_strategy):
        mapping = {
            getattr(self, name): getattr(concrete_phase_strategy, name)
            for name in self.maybe_symbols
            if isinstance(getattr(self, name), sp.Basic)
        }

        return mapping

    def generate_name_to_symbol_dict(self):
        return {name: getattr(self, name) for name in self.maybe_symbols}

    @classmethod
    def from_search_space_sample(
        cls,
        num_nodes: VarInt,
        num_gpus_per_node: VarInt,
        ckpt: VarInt,
        parallelism: Tuple[VarInt, VarInt, VarInt, VarInt, VarInt, VarInt],
        offloading: Tuple[VarFloat, VarFloat, VarFloat],
    ):
        per_device_batch_size, dp_size, tp_size, ws_size, gs_size, os_size = parallelism
        wo_ratio, go_ratio, oo_ratio = offloading
        return cls(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            ckpt=ckpt,
            per_device_batch_size=per_device_batch_size,
            dp_size=dp_size,
            tp_size=tp_size,
            ws_size=ws_size,
            gs_size=gs_size,
            os_size=os_size,
            wo_ratio=wo_ratio,
            go_ratio=go_ratio,
            oo_ratio=oo_ratio,
        )


def create_strategy_for_a_phase(name, config=None, mapping=None, layer_strategy=None):
    # Training
    per_device_batch_size = gsm.symbols(f"{name}_b", 1, integer=True, positive=True)

    # Parallelism
    dp_size = gsm.symbols(f"{name}_dp", 1, integer=True, positive=True)
    tp_size = gsm.symbols(f"{name}_tp", 1, integer=True, positive=True)
    ws_size = gsm.symbols(f"{name}_ws", 1, integer=True, positive=True)
    gs_size = gsm.symbols(f"{name}_gs", 1, integer=True, positive=True)
    os_size = gsm.symbols(f"{name}_os", 1, integer=True, positive=True)

    # Memory optimization
    wo_ratio = gsm.symbols(f"{name}_wo", 0.0, real=True, nonnegative=True)
    go_ratio = gsm.symbols(f"{name}_go", 0.0, real=True, nonnegative=True)
    oo_ratio = gsm.symbols(f"{name}_oo", 0.0, real=True, nonnegative=True)

    # e.g. input mapping is ``{"per_device_batch_size": b}``
    mapping = mapping or {}
    old_var2new_var = {}
    for new_var_str, old_var in mapping.items():
        new_var = eval(new_var_str)
        if isinstance(old_var, str):
            old_var2new_var[eval(old_var, gsm.name2symbol)] = new_var
        else:
            old_var2new_var[old_var] = new_var

    phase_strategy = PhaseStrategy(
        per_device_batch_size=per_device_batch_size,
        dp_size=dp_size,
        tp_size=tp_size,
        ws_size=ws_size,
        gs_size=gs_size,
        os_size=os_size,
        wo_ratio=wo_ratio,
        go_ratio=go_ratio,
        oo_ratio=oo_ratio,
        mapping=old_var2new_var,
    )

    if layer_strategy is not None:
        phase_strategy.layer_strategy = layer_strategy
        phase_strategy.num_nodes = layer_strategy.num_nodes
        phase_strategy.num_gpus_per_node = layer_strategy.num_gpus_per_node
        phase_strategy.ckpt = layer_strategy.ckpt

    return phase_strategy


def create_strategy_for_a_layer(name, config=None, mapping=None):
    num_nodes = gsm.symbols(f"{name}_m", 1, integer=True, positive=True)
    num_gpus_per_node = gsm.symbols(f"{name}_n", 1, integer=True, positive=True)
    ckpt = gsm.symbols(f"{name}_ckpt", 1, integer=True)
    if isinstance(mapping, (tuple, list)):
        assert len(mapping) == 2
        fwd_mapping, bwd_mapping = mapping
    else:
        fwd_mapping = mapping
        bwd_mapping = mapping

    layer_strategy = LayerStrategy(
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        ckpt=ckpt,
        share_strategy_for_fwd_bwd=True,
    )

    fwd_strategy = create_strategy_for_a_phase(
        f"{name}_fwd", config, fwd_mapping, layer_strategy=layer_strategy
    )
    bwd_strategy = create_strategy_for_a_phase(
        f"{name}_bwd", config, bwd_mapping, layer_strategy=layer_strategy
    )
    layer_strategy.fwd_strategy = fwd_strategy
    layer_strategy.bwd_strategy = bwd_strategy

    return layer_strategy


@dataclass
class DecisionSearchSpace:
    name: str
    search_space: Dict[str, Any]
    data_point2layer_strategy: Dict[Tuple[Any, ...], LayerStrategy]
    # mapping: Dict[int, LayerStrategy]
    # abstract_layer_strategy: LayerStrategy


@functools.cache
def create_decision_var_for_a_layer(
    name: str,
    layer_info,
    config: MistConfig,
    num_nodes: int,
    num_gpus_per_node: int,
    batch_size_per_micro_batch: int,
):
    """Create a decision variable for a layer.
    A decision var is used for searching. It can be encoded into an one-hot vector.
    Each element corresponds to a combination of a layer strategy.

    Parameters
    ----------
    name : str
        name of the decision variable
    layer_info : LayerInfo
        information for the layer
    config : MistConfig
        mist optimization configuration
    num_nodes : int
        number of nodes
    num_gpus_per_node : int
        number of gpus per node
    batch_size_per_micro_batch : int
        batch size per micro batch
    """

    num_gpus = num_nodes * num_gpus_per_node

    search_space = {}

    # CKPT
    if config.ckpt_tuning_enabled:
        ckpt_candidates = [True, False]
    else:
        ckpt_candidates = [True]
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

        # Sharding size is bounded by the number of GPUs
        ws_size_candidates = [n for n in POWER_OF_TWO if n <= dp_size]
        gs_size_candidates = [n for n in POWER_OF_TWO if n <= dp_size]
        os_size_candidates = [n for n in POWER_OF_TWO if n <= dp_size]

        for ws_size, gs_size, os_size in product(
            ws_size_candidates, gs_size_candidates, os_size_candidates
        ):
            parallelism_candidates.append(
                (batch_size, dp_size, tp_size, ws_size, gs_size, os_size)
            )

    # * Offloading
    wo_ratio_candidates = list(np.linspace(0.0, 1.0, 11))
    go_ratio_candidates = list(np.linspace(0.0, 1.0, 11))
    oo_ratio_candidates = list(np.linspace(0.0, 1.0, 11))

    if config.share_strategy_for_fwd_bwd:
        search_space["parallelism"] = parallelism_candidates
        search_space["wo_ratio"] = wo_ratio_candidates
        search_space["go_ratio"] = go_ratio_candidates
        search_space["oo_ratio"] = oo_ratio_candidates
    else:
        search_space["fwd_parallelism"] = parallelism_candidates
        search_space["fwd_wo_ratio"] = wo_ratio_candidates
        search_space["fwd_go_ratio"] = go_ratio_candidates
        search_space["fwd_oo_ratio"] = oo_ratio_candidates
        search_space["bwd_parallelism"] = parallelism_candidates
        search_space["bwd_wo_ratio"] = wo_ratio_candidates
        search_space["bwd_go_ratio"] = go_ratio_candidates
        search_space["bwd_oo_ratio"] = oo_ratio_candidates

    data_point2layer_strategy = {}

    # Get the corresponding info for each strategy
    for data_point in tqdm(product(*search_space.values())):
        (
            ckpt,
            (per_device_batch_size, dp_size, tp_size, ws_size, gs_size, os_size),
            wo_ratio,
            go_ratio,
            oo_ratio,
        ) = data_point
        _locals = locals()

        fwd_layer_info = layer_info[ExecType.FWD, ckpt]
        bwd_layer_info = layer_info[ExecType.BWD, ckpt]

        def convert_to_concrete_layer_info(info):
            strategy_symbol_mapping = {
                old_symbol: eval(name, _locals)
                for name, old_symbol in info.strategy.__dict__.items()
                if name != "mapping"
            }
            peak_memory = gsm.subs(info.peak_memory, strategy_symbol_mapping)
            saved_memory = gsm.subs(info.saved_memory, strategy_symbol_mapping)
            full_weights = gsm.subs(info.full_weights, strategy_symbol_mapping)
            sharded_and_offloaded_weights_in_gpu = gsm.subs(
                info.sharded_and_offloaded_weights_in_gpu, strategy_symbol_mapping
            )
            full_grads = gsm.subs(info.full_grads, strategy_symbol_mapping)
            sharded_and_offloaded_grads_in_gpu = gsm.subs(
                info.sharded_and_offloaded_grads_in_gpu, strategy_symbol_mapping
            )
            full_opts = gsm.subs(info.full_opts, strategy_symbol_mapping)
            sharded_and_offloaded_opts_in_gpu = gsm.subs(
                info.sharded_and_offloaded_opts_in_gpu, strategy_symbol_mapping
            )
            return LayerInfo(
                peak_memory=peak_memory,
                saved_memory=saved_memory,
                full_weights=full_weights,
                sharded_and_offloaded_weights_in_gpu=sharded_and_offloaded_weights_in_gpu,
                full_grads=full_grads,
                sharded_and_offloaded_grads_in_gpu=sharded_and_offloaded_grads_in_gpu,
                full_opts=full_opts,
                sharded_and_offloaded_opts_in_gpu=sharded_and_offloaded_opts_in_gpu,
            )

        concrete_fwd_layer_info = convert_to_concrete_layer_info(fwd_layer_info)
        concrete_bwd_layer_info = convert_to_concrete_layer_info(bwd_layer_info)

        data_point2layer_strategy[data_point] = (
            concrete_fwd_layer_info,
            concrete_bwd_layer_info,
        )

    return DecisionSearchSpace(
        name=name,
        search_space=search_space,
        data_point2layer_strategy=data_point2layer_strategy,
    )


def create_decision_vars_for_a_stage_module(
    block_layer_partitions,
    gradient_accumulation_steps,
    config,
    has_pre_layer=True,
    has_post_layer=True,
):
    """
    Generate the search space and corresponding symbols for the general pipeline.

    Parameters
    ----------
    layer_partitions: List[int]
        a sequence denoting the number of block layers in each stages
    config : MistConfig
        mist optimization configuration
    """
    assert sum(block_layer_partitions) == config.num_layers

    num_gpus = config.num_nodes * config.num_gpus_per_node
    num_layers = config.num_layers
    total_num_layers = config.num_layers + 2
    num_stages = len(block_layer_partitions)
    num_block_layers_per_stage = [len(stage) for stage in block_layer_partitions]
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
    create_decision_var_for_a_layer = partial(
        create_decision_var_for_a_layer,
        config=config,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        batch_size_per_micro_batch=batch_size_per_micro_batch,
    )

    unique_layer_decision_vars = []

    # pre- and post- layers
    if config.strategy_granularity in {"model", "stage", "layer"}:
        pre_var = create_decision_var_for_a_layer("pre_layer")
        post_var = create_decision_var_for_a_layer("post_layer")
        unique_layer_decision_vars.append(pre_var)
        unique_layer_decision_vars.append(post_var)
    elif config.strategy_granularity in {"micro_batch", "phase"}:
        # Pre-layer can only show up in the first stage
        # which should have (num_warmup_phases + 1 (1F1B)) phases
        for i in range(num_stages):
            pre_var = create_decision_var_for_a_layer(f"pre_layer_mb{i}")
            unique_layer_decision_vars.append(pre_var)
        # Post-layer can only show up in the last stage
        # which should have 1 (1F1B) phase
        post_var = create_decision_var_for_a_layer("post_layer")
        unique_layer_decision_vars.append(post_var)
    else:
        raise ValueError(f"Unknown strategy_granularity: {config.strategy_granularity}")

    # block layers
    if config.strategy_granularity == "pipe":
        # All phases share the same strategy
        block_var = create_decision_var_for_a_layer("block_layer")
        unique_layer_decision_vars.append(block_var)
    elif config.strategy_granularity == "stage":
        # All phases in a stage share the same strategy
        for stage_idx in range(num_stages):
            block_var = create_decision_var_for_a_layer(f"block_layer_stage{stage_idx}")
            unique_layer_decision_vars.append(block_var)
    elif config.strategy_granularity == "micro_batch":
        for stage_idx in range(num_stages):
            warmup, fb = calculate_num_warmup_and_1f1b_phases(
                stage_idx, num_stages, gradient_accumulation_steps
            )
            for mb_idx in range(warmup + 1):
                block_var = create_decision_var_for_a_layer(
                    f"block_layer_stage{stage_idx}_mb{mb_idx}"
                )
                unique_layer_decision_vars.append(block_var)
    elif config.strategy_granularity == "layer":
        for stage_idx in range(num_stages):
            for layer_idx in range(num_block_layers_per_stage[stage_idx]):
                block_var = create_decision_var_for_a_layer(
                    f"block_layer_stage{stage_idx}_layer{layer_idx}"
                )
                unique_layer_decision_vars.append(block_var)
    elif config.strategy_granularity == "phase":
        for stage_idx in range(num_stages):
            warmup, fb = calculate_num_warmup_and_1f1b_phases(
                stage_idx, num_stages, gradient_accumulation_steps
            )
            for mb_idx in range(warmup + 1):
                for layer_idx in range(num_block_layers_per_stage[stage_idx]):
                    block_var = create_decision_var_for_a_layer(
                        f"block_layer_stage{stage_idx}_layer{layer_idx}_mb{mb_idx}",
                        config,
                    )
                    unique_layer_decision_vars.append(block_var)
    else:
        raise ValueError(f"Unknown strategy_granularity: {config.strategy_granularity}")
