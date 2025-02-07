import itertools
import os
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, cache
from itertools import product
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
from mist.logger import get_logger
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
from mist.utils.common import load_pickle, save_pickle
from mist.utils.memory import cuda_empty_cache
from mist.utils.pipeline_parallel import calculate_num_warmup_and_1f1b_phases
from mist.utils.sympy import autowrap_with_cython, ufuncify_with_cython
from mist.analyzer.batched_module_analyzer import batched_tune_best_latency_for_stage, SAMPLE_SIZE

logger = get_logger()

SAVE_INTRA_TUNING_RESULTS = False

def _np_set_item(array, value):
    """Set value (a small array) to array (a large array)

    It will put the small array to the upper left corner of the large array.
    """
    assert array.shape >= value.shape
    indicies = tuple(slice(0, v) for v in value.shape)
    array[indicies] = value
    return array


@dataclass(frozen=True)
class IntraStageTunerOutput:
    gradient_accumulation_steps_candidates: List[int]
    device_mesh_candidates: List[Tuple[int, int]]
    costs_stable: np.ndarray
    costs_stable_with_pre: np.ndarray
    costs_stable_with_post: np.ndarray
    costs_stable_no_pp: np.ndarray
    costs_delta: np.ndarray
    costs_delta_with_pre: np.ndarray
    costs_delta_with_post: np.ndarray
    costs_delta_no_pp: np.ndarray
    solutions: np.ndarray
    solutions_with_pre: np.ndarray
    solutions_with_post: np.ndarray
    solutions_no_pp: np.ndarray

    def __repr__(self) -> str:
        return (
            f"IntraStageTunerOutput("
            f"gradient_accumulation_steps_candidates={self.gradient_accumulation_steps_candidates}, "
            f"device_mesh_candidates={self.device_mesh_candidates}."
        )


class IntraStageTuner:
    def __init__(
        self,
        block_layer_info: LayerInfo,
        pre_layer_info: LayerInfo,
        post_layer_info: LayerInfo,
        num_layers: int,
        device_mesh_candidates: List[Tuple[int, int]],
        gradient_accumulation_steps_candidates: List[int],
        tp_size: int,
        config: MistConfig,
        disable_tqdm: bool = False,
        sample_size: int = SAMPLE_SIZE,
    ):
        self.block_layer_info = block_layer_info
        self.pre_layer_info = pre_layer_info
        self.post_layer_info = post_layer_info
        self.num_layers = num_layers
        self.device_mesh_candidates = device_mesh_candidates
        self.gradient_accumulation_steps_candidates = (
            gradient_accumulation_steps_candidates
        )
        self.tp_size = tp_size
        self.config = config
        self.disable_tqdm = disable_tqdm
        self.sample_size = sample_size

        self.device_mesh_to_index = {q: i for i, q in enumerate(device_mesh_candidates)}
        self.gradient_accumulation_steps_to_index = {
            g: i for i, g in enumerate(gradient_accumulation_steps_candidates)
        }

        self.max_possible_num_stages = min(
            max(n * m for n, m in device_mesh_candidates), num_layers
        )
        self.max_possible_pre_saved_micro_batches = (
            min(
                self.max_possible_num_stages,
                max(gradient_accumulation_steps_candidates),
            )
            - 1
        )

        # ------------------------------
        # n: num_nodes
        # m: num_gpus_per_node
        # q: device_mesh_candidate_index
        # g: gradient_accumulation_steps
        # ------------------------------
        # p: pre_saved_micro_batches (0-indexed)
        # l: num_layers (1-indexed)
        # c: num_ckpt_layers (0-indexed)
        # f: feature_size
        # ------------------------------
        # costs     / costs_with_pre:     (g, q) -> (p, l, c)
        # solutions / solutions_with_pre: (g, q) -> (p, l, c, f))
        # ------
        # costs_with_post:     (g, q) -> (l, c)
        # solutions_with_post: (g, q) -> (l, c, f)
        # * Note: no need to have `p` because `p` is always 0 (for the last stage)
        # ------
        # costs_no_pp:     (g, q) -> (c)
        # solutions_no_pp: (g, q) -> (c, f)
        # * Note: no need to have `p` and `l` because `p` is always 0
        # * and `l` is always `c + 1`
        g = len(self.gradient_accumulation_steps_candidates)
        q = len(self.device_mesh_candidates)
        p = self.max_possible_pre_saved_micro_batches
        l = self.num_layers
        s = self.sample_size
        f = 10
        self._costs_stable = np.full(((g, q, p + 1, l, l + 1, s)), np.inf)
        self._costs_stable_with_pre = np.full(((g, q, p + 1, l, l + 1, s)), np.inf)
        self._costs_stable_with_post = np.full(((g, q, l, l + 1, s)), np.inf)
        self._costs_stable_no_pp = np.full(((g, q, l + 1, s)), np.inf)
        self._costs_delta = np.full(((g, q, p + 1, l, l + 1, s)), np.inf)
        self._costs_delta_with_pre = np.full(((g, q, p + 1, l, l + 1, s)), np.inf)
        self._costs_delta_with_post = np.full(((g, q, l, l + 1, s)), np.inf)
        self._costs_delta_no_pp = np.full(((g, q, l + 1, s)), np.inf)
        self._solutions = np.full(((g, q, p + 1, l, l + 1, s, f)), np.inf)
        self._solutions_with_pre = np.full(((g, q, p + 1, l, l + 1, s, f)), np.inf)
        self._solutions_with_post = np.full(((g, q, l, l + 1, s, f)), np.inf)
        self._solutions_no_pp = np.full(((g, q, l + 1, s, f)), np.inf)
        self.results = None

    def tune(self) -> IntraStageTunerOutput:
        for g, gradient_accumulation_steps in tqdm(
            enumerate(self.gradient_accumulation_steps_candidates),
            desc="Intra-stage Tuning GradAccu",
            total=len(self.gradient_accumulation_steps_candidates),
            disable=self.disable_tqdm,
            position=0,
        ):
            for q, (num_nodes, num_gpus_per_node) in tqdm(
                enumerate(self.device_mesh_candidates),
                desc="Intra-stage Tuning DeviceMesh",
                total=len(self.device_mesh_candidates),
                disable=self.disable_tqdm,
                position=1,
                leave=False,
            ):
                pre_saved_micro_batches_candidates = list(
                    range(
                        min(
                            gradient_accumulation_steps,
                            self.num_layers,
                            self.max_possible_pre_saved_micro_batches + 1,
                        )
                    )
                )
                num_layers_candidates = list(range(1, self.num_layers + 1))
                num_ckpt_layers_candidates = list(range(0, self.num_layers + 1))
                (
                    (stage_costs_stable, stage_costs_delta, stage_solutions),
                    (
                        stage_costs_stable_with_pre,
                        stage_costs_delta_with_pre,
                        stage_solutions_with_pre,
                    ),
                    (
                        stage_costs_stable_with_post,
                        stage_costs_delta_with_post,
                        stage_solutions_with_post,
                    ),
                    (
                        stage_costs_stable_no_pp,
                        stage_costs_delta_no_pp,
                        stage_solutions_no_pp,
                    ),
                ) = batched_tune_best_latency_for_stage(
                    block_layer_info=self.block_layer_info,
                    pre_layer_info=self.pre_layer_info,
                    post_layer_info=self.post_layer_info,
                    pre_saved_micro_batches_candidates=pre_saved_micro_batches_candidates,
                    num_layers_candidates=num_layers_candidates,
                    num_ckpt_layers_candidates=num_ckpt_layers_candidates,
                    num_nodes=num_nodes,
                    num_gpus_per_node=num_gpus_per_node,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    sample_size=self.sample_size,
                    tp_size=self.tp_size,
                    config=self.config,
                )

                _np_set_item(self._costs_stable[g, q], stage_costs_stable)
                _np_set_item(
                    self._costs_stable_with_pre[g, q], stage_costs_stable_with_pre
                )
                _np_set_item(
                    self._costs_stable_with_post[g, q], stage_costs_stable_with_post
                )
                _np_set_item(self._costs_stable_no_pp[g, q], stage_costs_stable_no_pp)
                _np_set_item(self._costs_delta[g, q], stage_costs_delta)
                _np_set_item(
                    self._costs_delta_with_pre[g, q], stage_costs_delta_with_pre
                )
                _np_set_item(
                    self._costs_delta_with_post[g, q], stage_costs_delta_with_post
                )
                _np_set_item(self._costs_delta_no_pp[g, q], stage_costs_delta_no_pp)
                self._costs_stable = self._costs_stable.astype(np.float32)
                self._costs_stable_with_pre = self._costs_stable_with_pre.astype(
                    np.float32
                )
                self._costs_stable_with_post = self._costs_stable_with_post.astype(
                    np.float32
                )
                self._costs_stable_no_pp = self._costs_stable_no_pp.astype(np.float32)
                self._costs_delta = self._costs_delta.astype(np.float32)
                self._costs_delta_with_pre = self._costs_delta_with_pre.astype(
                    np.float32
                )
                self._costs_delta_with_post = self._costs_delta_with_post.astype(
                    np.float32
                )
                self._costs_delta_no_pp = self._costs_delta_no_pp.astype(np.float32)
                _np_set_item(self._solutions[g, q], stage_solutions)
                _np_set_item(self._solutions_with_pre[g, q], stage_solutions_with_pre)
                _np_set_item(self._solutions_with_post[g, q], stage_solutions_with_post)
                _np_set_item(self._solutions_no_pp[g, q], stage_solutions_no_pp)

        self.results = IntraStageTunerOutput(
            gradient_accumulation_steps_candidates=self.gradient_accumulation_steps_candidates,
            device_mesh_candidates=self.device_mesh_candidates,
            costs_stable=self._costs_stable,
            costs_stable_with_pre=self._costs_stable_with_pre,
            costs_stable_with_post=self._costs_stable_with_post,
            costs_stable_no_pp=self._costs_stable_no_pp,
            costs_delta=self._costs_delta,
            costs_delta_with_pre=self._costs_delta_with_pre,
            costs_delta_with_post=self._costs_delta_with_post,
            costs_delta_no_pp=self._costs_delta_no_pp,
            solutions=self._solutions,
            solutions_with_pre=self._solutions_with_pre,
            solutions_with_post=self._solutions_with_post,
            solutions_no_pp=self._solutions_no_pp,
        )

        return self.results


def intra_stage_tune(
    block_layer_info: LayerInfo,
    pre_layer_info: LayerInfo,
    post_layer_info: LayerInfo,
    num_layers: int,
    device_mesh_candidates: List[Tuple[int, int]],
    gradient_accumulation_steps_candidates: List[int],
    tp_size: int,
    mist_config: MistConfig,
    disable_tqdm: bool = False,
    saved_path: str = None,
    force_recompute: bool = False,
    sample_size: int = SAMPLE_SIZE,
):
    if saved_path is None:
        model_config = mist_config.model
        training_config = mist_config.training
        strategy_config = mist_config.strategy
        hardware_config = mist_config.hardware
        tuning_config = mist_config.tuning
        if tuning_config.tuning_granularity in [
            "uniform-pp",
            "uniform-device-pp",
            "uniform-device-pp-mip",
            "inter-stage",
            "uniform-pp-simple-heuristic-mem-opt",
        ]:
            tuning_granularity_str = "pp"
        elif tuning_config.tuning_granularity == "no-pp":
            tuning_granularity_str = "no-pp"
        else:
            raise ValueError(
                f"Unknown tuning_granularity: {tuning_config.tuning_granularity}"
            )
        saved_folder = "intra_stage_tuning_results"
        if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)
        saved_path = (
            f"intra_stage_tuning_results_"
            f"model_{model_config.name}_layer_{num_layers}_"
            f"b_{training_config.global_batch_size}_"
            f"n_{hardware_config.num_nodes}_m_{hardware_config.num_gpus_per_node}_"
            f"flash_{model_config.use_flash_attn}_"
            f"bw_{hardware_config.gpu_gpu_comm_params[3]:.2f}_"
            f"gc_bw_{hardware_config.gpu_cpu_comm_params[0]:.2f}_"
            f"cg_bw_{hardware_config.cpu_gpu_comm_params[0]:.2f}_"
            f"mem_{hardware_config.memory_capacity:.1f}_"
            f"tuning_{tuning_granularity_str}_"
            f"sample_{sample_size}_"
            f"zero23_{tuning_config.zero_2_and_3_enabled}_"
            f"ac_{tuning_config.activation_checkpointing_tuning_enabled}_"
            f"so_{tuning_config.state_offloading_enabled}_"
            f"ao_{tuning_config.activation_offloading_enabled}_"
            f"tp_{tp_size}_"
            f"fixed_ao_{getattr(mist_config, 'fixed_ao_ratio', 'None')}_"
            f"fixed_wo_{getattr(mist_config, 'fixed_wo_ratio', 'None')}_"
            f"fixed_go_{getattr(mist_config, 'fixed_go_ratio', 'None')}_"
            f"fixed_oo_{getattr(mist_config, 'fixed_oo_ratio', 'None')}_"
            f"pre_post_{','.join([str(p) for p in tuning_config.pre_post_strategy_array])}.pkl"
        )
        saved_path = os.path.join(saved_folder, saved_path)
    saved_path = saved_path.lower()

    if not force_recompute and os.path.exists(saved_path):
        intra_stage_results = load_pickle(saved_path)
        logger.info(f"Loading saved intra-stage tuning results from {saved_path}")
    else:
        # Intra-stage strategy tuning
        intra_stage_tuner = IntraStageTuner(
            block_layer_info=block_layer_info,
            pre_layer_info=pre_layer_info,
            post_layer_info=post_layer_info,
            num_layers=num_layers,
            device_mesh_candidates=device_mesh_candidates,
            gradient_accumulation_steps_candidates=gradient_accumulation_steps_candidates,
            tp_size=tp_size,
            config=mist_config,
            disable_tqdm=disable_tqdm,
            sample_size=sample_size,
        )
        intra_stage_results = intra_stage_tuner.tune()

        if SAVE_INTRA_TUNING_RESULTS:
            save_pickle(intra_stage_results, saved_path)

    return intra_stage_results


# def _trace_analyze_and_intra_stage_tune(
#     layer_info_provider,
#     num_layers: int,
#     device_mesh_candidates: List[Tuple[int, int]],
#     gradient_accumulation_steps_candidates: List[int],
#     mist_config: MistConfig,
#     disable_tqdm: bool = False,
#     saved_path: str = None,
#     force_recompute: bool = False,
# ):
#     """It's a tmp function for fast iteration"""

#     if saved_path is None:
#         model_config = mist_config.model
#         training_config = mist_config.training
#         strategy_config = mist_config.strategy
#         hardware_config = mist_config.hardware
#         tuning_config = mist_config.tuning
#         saved_path = (
#             f"intra_stage_tuning_results_"
#             f"model_{model_config.name}_layer_{num_layers}_"
#             f"b_{training_config.global_batch_size}_"
#             f"n_{hardware_config.num_nodes}_m_{hardware_config.num_gpus_per_node}_"
#             f"inter_bw_{hardware_config.inter_node_gpu_gpu_bandwidth}_"
#             f"intra_bw_{hardware_config.intra_node_gpu_gpu_bandwidth}_"
#             f"gc_bw_{hardware_config.gpu_cpu_bandwidth}_"
#             f"mem_{hardware_config.memory_capacity}_"
#             f"tuning_{tuning_config.tuning_granularity}_"
#             f"so_{tuning_config.state_offloading_enabled}_"
#             f"ao_{tuning_config.activation_offloading_enabled}.pkl"
#         )

#     if not force_recompute and os.path.exists(saved_path):
#         intra_stage_results = load_pickle(saved_path)
#     else:
#         # Trace and analyze
#         layer_infos = layer_info_provider()
#         cuda_empty_cache()

#         # Intra-stage strategy tuning
#         intra_stage_tuner = IntraStageTuner(
#             block_layer_info=layer_infos["block_layer"],
#             pre_layer_info=layer_infos["pre_layer"],
#             post_layer_info=layer_infos["post_layer"],
#             num_layers=num_layers,
#             device_mesh_candidates=device_mesh_candidates,
#             gradient_accumulation_steps_candidates=gradient_accumulation_steps_candidates,
#             config=mist_config,
#             disable_tqdm=disable_tqdm,
#         )
#         intra_stage_results = intra_stage_tuner.tune()

#         save_pickle(intra_stage_results, saved_path)

#     return intra_stage_results
