from typing import Optional, Dict, List, Tuple, Any

import sympy as sp

from mist import global_symbol_manager as gsm
from mist.analyzer.layer_analyzer import LayerInfo
from mist.analyzer.recorder import ExecType
from mist.analyzer.strategy import ModelStrategy, LayerStrategy
from mist.config import MistConfig
from mist.utils.pipeline_parallel import calculate_num_warmup_and_1f1b_phases


def parse_layer_combination(layer_combination):
    layers2nums = {}
    for combination_str in layer_combination:
        assert combination_str.count("*") <= 1
        if combination_str.count("*") == 0:
            layer_name = combination_str
            num = 1
        else:
            layer_name, num = combination_str.split("*")
            print(layer_name, num)
            try:
                num = int(num)
            except ValueError:
                num = eval(num, gsm.name2symbol)
        layers2nums[layer_name] = num
    return layers2nums


def calculate_states(
    params_status, grads_status, opt_status, fwd_info, layer_strategies
):
    assert (
        num_layers := len(params_status)
        == len(grads_status)
        == len(opt_status)
        == len(fwd_info)
        == len(layer_strategies)
    )

    def map_to_state(status, item, info):
        assert item in {"weights", "grads", "opts"}
        if status == "F":
            attr_name = f"full_{item}"
        elif status == "SO":
            attr_name = f"sharded_and_offloaded_{item}_in_gpu"
        else:
            raise ValueError(f"Invalid status: {status}")
        return getattr(info, attr_name)

    states = 0
    for i in range(num_layers):
        cur_state = 0
        cur_state += map_to_state(params_status[i], "weights", fwd_info[i])
        cur_state += map_to_state(grads_status[i], "grads", fwd_info[i])
        cur_state += map_to_state(opt_status[i], "opts", fwd_info[i])
        states += gsm.subs(cur_state, layer_strategies[i].mapping)
    return states


def create_memory_expr_for_warmup(
    layer_infos: Dict[str, LayerInfo],
    layer_combination: List[str],
    layer_strategies: List[LayerStrategy],
    num_phase: int,
):
    """
    layer_infos: Dict[str, Dict[str, LayerInfo]]
    layer_combination: layer combination for each phase
    layer_strategies: layer strategies for each phase * layer
    num_phase: number of phases of warmup
    """

    # Check the validity of layer_combination
    assert all(layer in layer_infos for layer in layer_combination)
    # Check the validity of layer_strategies, or use the default ones
    assert len(layer_strategies) == len(layer_combination) * (num_phase + 1)

    """
    Wramup only runs the forward pass iteratively, e.g.:
    [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4] for 3 phases and 4 layers.

    When running the layer n - 1, layer 0 will be fully loaded into the memory. 
    """
    params_status_for_each_layer = ["F", "F"] + ["SO"] * (len(layer_combination) - 2)
    grads_status_for_each_layer = ["SO"] * len(layer_combination)
    opt_states_status_for_each_layer = ["SO"] * len(layer_combination)

    states_for_each_layer = []
    states_all = None
    saved_tensors = 0
    peak_memory = 0

    fwd_info_ckpts = []
    fwd_info_no_ckpts = []
    fwd_strategies = [s.strategies[0] for s in layer_strategies]
    num_layer_per_phase = len(layer_combination)
    num = len(layer_combination) * num_phase
    for i, layer_strategy in enumerate(layer_strategies):
        layer_name = layer_combination[i % num_layer_per_phase]
        fwd_info_ckpt = layer_infos[layer_name][ExecType.FWD, True]
        fwd_info_ckpt.calculate_aux()
        fwd_info_ckpts.append(fwd_info_ckpt)
        fwd_info_no_ckpt = layer_infos[layer_name][ExecType.FWD, False]
        fwd_info_no_ckpt.calculate_aux()
        fwd_info_no_ckpts.append(fwd_info_no_ckpt)

    for i in range(num):
        layer_name = layer_combination[i % num_layer_per_phase]
        layer_strategy = layer_strategies[i]
        ckpt = layer_strategy.ckpt
        fwd_strategy = layer_strategy.strategies[0]

        # Memory states
        memory_states = calculate_states(
            params_status_for_each_layer,
            grads_status_for_each_layer,
            opt_states_status_for_each_layer,
            fwd_info_ckpts[i : i + num_layer_per_phase],
            fwd_strategies[i : i + num_layer_per_phase],
        )

        # Peak intermediate memory
        peak_intermediate_memory_ckpt = gsm.subs(
            fwd_info_ckpts[i].peak_memory, fwd_strategy.mapping
        )
        peak_intermediate_memory_no_ckpt = gsm.subs(
            fwd_info_no_ckpts[i].peak_memory, fwd_strategy.mapping
        )
        cur_peak_memory = (
            sp.Piecewise(
                (peak_intermediate_memory_no_ckpt, sp.Eq(ckpt, 0)),
                (peak_intermediate_memory_ckpt, sp.Eq(ckpt, 1)),
                (1e12, True),
            )
            + saved_tensors
            + memory_states
        )
        peak_memory = sp.Max(peak_memory, cur_peak_memory)

        # Saved tensors
        cur_saved_tensors_ckpt = gsm.subs(
            fwd_info_ckpts[i].saved_memory, fwd_strategy.mapping
        )
        cur_saved_tensors_no_ckpt = gsm.subs(
            fwd_info_no_ckpts[i].saved_memory, fwd_strategy.mapping
        )
        saved_tensors = (
            sp.Piecewise(
                (cur_saved_tensors_no_ckpt, sp.Eq(ckpt, 0)),
                (cur_saved_tensors_ckpt, sp.Eq(ckpt, 1)),
                (1e12, True),
            )
            + saved_tensors
        )

    return peak_memory, saved_tensors

    print("1")


class ModelAnalyzer:
    def __init__(
        self,
        config: MistConfig,
        gradient_accumulation_steps: int,
        block_layer_info: LayerInfo,
        block_layer_partition: Optional[List[int]] = None,
        pre_layer_info: Optional[LayerInfo] = None,
        post_layer_info: Optional[LayerInfo] = None,
    ):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.block_layer_info = block_layer_info
        self.block_layer_partition = block_layer_partition
        self.num_stages = len(block_layer_partition) if block_layer_partition else 1
        self.pre_layer_info = pre_layer_info
        self.post_layer_info = post_layer_info

    def run_concrete(self, model_strategy: ModelStrategy):
        peaks_memories = []
        strategy_granularity = model_strategy.strategy_granularity
        if strategy_granularity in {"model", "stage"}:
            for stage_idx in range(self.num_stages):
                cur_peak_memories = self._run_concrete_1f1b(stage_idx=stage_idx)
                peaks_memories.extend(cur_peak_memories)
        elif strategy_granularity == "micro_batch":
            for stage_idx in range(self.num_stages):
                warmup, _ = calculate_num_warmup_and_1f1b_phases(
                    stage_idx, self.num_stages, self.gradient_accumulation_steps
                )
                cur_peak_memories = self._run_concrete_1f1b(stage_idx=stage_idx)
                peaks_memories.extend(cur_peak_memories)
                for warmup_idx in range(warmup):
                    cur_peak_memories = self._run_concrete_warmup(
                        stage_idx=stage_idx, warmup_idx=warmup_idx
                    )
                    peaks_memories.extend(cur_peak_memories)
                    cur_peak_memories = self._run_concrete_cooldown(
                        stage_idx=stage_idx, cooldown_idx=warmup_idx
                    )
                    peaks_memories.extend(cur_peak_memories)
        else:
            raise NotImplementedError

        latency = 1

        return peaks_memories, latency

    def _run_concrete_1f1b(self, stage_idx: int):
        return [1]

    def _run_concrete_warmup(self, stage_idx: int, warmup_idx: int):
        return [1]

    def _run_concrete_cooldown(self, stage_idx: int, cooldown_idx: int):
        return [1]
