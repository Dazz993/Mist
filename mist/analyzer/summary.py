"""Summary of the analysis results."""
from __future__ import annotations
from typing import List, Optional, Tuple, Union, Dict, TYPE_CHECKING
from itertools import product
from collections import OrderedDict
import os
import json

import torch
from torch.utils._pytree import tree_map

import numbers
import sympy as sp

from mist.memory_pool import (
    MemorySnapshot,
    remove_weights_in_saved_tensors,
    remove_weights_in_saved_tensors_in_snapshot,
    peak_memory_among_different_snapshots,
    compute_memory_for_flattened,
    nbytes,
)
from mist.node_database.database import NodeDB
from mist.node_database.node_spec import NodeSpec
from mist.node_database.symbolic_node_spec import SymbolicNodeSpec
from mist.utils.common import save_json, pprint_to_file
from mist.utils.sympy import sp2py, floor_div

if TYPE_CHECKING:
    from mist.analyzer.recorder import MistSymbolicAnalyzer

# db = NodeDB()


def get_params_and_buffers(module):
    params = 0
    buffers = 0
    for p in module.parameters():
        if p.requires_grad:
            params += nbytes(p)
        else:
            buffers += nbytes(p)
    for b in module.buffers():
        buffers += nbytes(b)
    return params, buffers


class MemorySummarizer:
    def __init__(self, analyzer: MistSymbolicAnalyzer):
        self.analyzer = analyzer
        self.stage_info = OrderedDict({stage: {} for stage in self.analyzer.stages})

        for stage in self.analyzer.stages:
            self.stage_summary(stage)

    def stage_summary(self, stage_name):
        """Summary of the basic stage information.

        - In the forward pass, the intermediate memory of the stage = saved tensors from the previous stage + peak intermediate memory of the stage.
        - In the backward pass, the intermediate memory of the stage = saved tensors from the previous stage (the order is determined in the forward pass) + peak intermediate memory of the stage.
        """

        analyzer = self.analyzer.stage_analyzer[stage_name]

        ## States
        module = self.analyzer.stage_module[stage_name]
        config = self.analyzer.stage_config[stage_name]
        wgt_fwd_rdn_sharding_size = config.get("wgt_fwd_rdn_sharding_size")
        wgt_bwd_rdn_sharding_size = config.get("wgt_bwd_rdn_sharding_size")
        wgt_fwd_ofd_ratio = config.get("wgt_fwd_ofd_ratio")
        wgt_bwd_ofd_ratio = config.get("wgt_bwd_ofd_ratio")
        grad_ofd_ratio = config.get("grad_ofd_ratio")  # No sharding for grad
        opt_rdn_sharding_size = config.get("opt_rdn_sharding_size")
        opt_ofd_ratio = config.get("opt_ofd_ratio")

        params, buffers = get_params_and_buffers(module)
        # Weights
        full_weights = params + buffers
        sharded_and_offloaded_weights_in_gpu_fwd = (
            full_weights / wgt_fwd_rdn_sharding_size * (1 - wgt_fwd_ofd_ratio)
        )
        sharded_and_offloaded_weights_in_gpu_bwd = (
            full_weights / wgt_bwd_rdn_sharding_size * (1 - wgt_bwd_ofd_ratio)
        )
        # Grads
        full_grads = full_weights
        offloaded_grads_in_gpu = full_weights * (1 - grad_ofd_ratio)
        # Optimizer states
        full_opt_states = full_weights * 2  # TODO(zhanda): 2 is hardcoded for fp32 adam
        sharded_and_offloaded_opt_in_gpu = (
            full_opt_states / opt_rdn_sharding_size * (1 - opt_ofd_ratio)
        )

        ## Intermediate memory
        # CKPT
        ckpt_saved_tensors = set(
            analyzer.ckpt_node2info[analyzer.output_node]
            .snapshot_before_release.get_category("saved_tensors")
            .keys()
        )
        ckpt_peak_inside_stage_fwd = peak_memory_among_different_snapshots(
            snapshots=[
                info.snapshot_before_release
                for info in analyzer.ckpt_node2info.values()
            ],
            categories=["intermediate", "saved_tensors"],
        )

        # FWD
        fwd_saved_tensors = set(
            analyzer.fwd_node2info[analyzer.output_node]
            .snapshot_before_release.get_category("saved_tensors")
            .keys()
        )
        fwd_peak_inside_stage_fwd = peak_memory_among_different_snapshots(
            snapshots=[
                info.snapshot_before_release for info in analyzer.fwd_node2info.values()
            ],
            categories=["intermediate", "saved_tensors"],
        )

        # BWD
        bwd_peak_inside_stage_bwd = peak_memory_among_different_snapshots(
            snapshots=[
                info.snapshot_before_release for info in analyzer.bwd_node2info.values()
            ],
            categories=["intermediate", "saved_tensors"],
        )

        # Update the stage info
        self.stage_info[stage_name].update(
            # States
            full_weights=full_weights,
            full_grads=full_grads,
            full_opt_states=full_opt_states,
            sharded_and_offloaded_weights_in_gpu_fwd=sharded_and_offloaded_weights_in_gpu_fwd,
            sharded_and_offloaded_weights_in_gpu_bwd=sharded_and_offloaded_weights_in_gpu_bwd,
            offloaded_grads_in_gpu=offloaded_grads_in_gpu,
            sharded_and_offloaded_opt_in_gpu=sharded_and_offloaded_opt_in_gpu,
            # Intermediate memory
            ckpt_saved_tensors=ckpt_saved_tensors,
            ckpt_peak_inside_stage_fwd=ckpt_peak_inside_stage_fwd,
            fwd_saved_tensors=fwd_saved_tensors,
            fwd_peak_inside_stage_fwd=fwd_peak_inside_stage_fwd,
            bwd_peak_inside_stage_bwd=bwd_peak_inside_stage_bwd,
        )

    def get_current_state_for_stage_layer(
        self, stage_name, exec_type, weights_full=True, grad_full=True, opt_full=True
    ):
        info = self.stage_info[stage_name]
        if not exec_type in {"fwd", "bwd"}:
            raise ValueError(f"Unknown exec_type {exec_type}")

        states = 0
        states += (
            info["full_weights"]
            if weights_full
            else info[f"sharded_and_offloaded_weights_in_gpu_{exec_type}"]
        )
        states += info["full_grads"] if grad_full else info["offloaded_grads_in_gpu"]
        states += (
            info["full_opt_states"]
            if opt_full
            else info["sharded_and_offloaded_opt_in_gpu"]
        )
        return states

    def get_current_states_for_model_fwd(self, stage_name, block_idx=0):
        states = 0
        n_block_layers = self.analyzer.stage_config["block"].get("n_layers")

        has_preprocesing = "preprocessing" in self.analyzer.stages
        has_postprocessing = "postprocessing" in self.analyzer.stages

        if stage_name == "preprocessing":
            states += self.get_current_state_for_stage_layer(
                "preprocessing", "fwd", True, False, False
            )
            states += self.get_current_state_for_stage_layer(
                "block", "fwd", True, False, False
            )
            states += self.get_current_state_for_stage_layer(
                "block", "fwd", False, False, False
            ) * (n_block_layers - 1)
            if has_postprocessing:
                states += self.get_current_state_for_stage_layer(
                    "postprocessing", "fwd", False, False, False
                )

        elif stage_name == "block":
            if has_preprocesing:
                states += self.get_current_state_for_stage_layer(
                    "preprocessing", "bwd", False, False, False
                )

            if block_idx <= n_block_layers - 2:
                states += (
                    self.get_current_state_for_stage_layer(
                        "block", "bwd", False, False, False
                    )
                    * block_idx
                )
                states += (
                    self.get_current_state_for_stage_layer(
                        "block", "fwd", True, False, False
                    )
                    * 2
                )
                states += self.get_current_state_for_stage_layer(
                    "block", "bwd", False, False, False
                ) * (n_block_layers - block_idx - 2)
                if has_postprocessing:
                    states += self.get_current_state_for_stage_layer(
                        "postprocessing", "fwd", False, False, False
                    )
            elif not has_postprocessing:
                states += self.get_current_state_for_stage_layer(
                    "block", "bwd", False, False, False
                ) * (n_block_layers - 2)
                states += self.get_current_state_for_stage_layer(
                    "block", "bwd", True, False, False
                )
                states += self.get_current_state_for_stage_layer(
                    "block", "fwd", True, False, False
                )
            else:
                states += self.get_current_state_for_stage_layer(
                    "block", "bwd", False, False, False
                ) * (n_block_layers - 1)
                states += self.get_current_state_for_stage_layer(
                    "block", "fwd", True, False, False
                )
                states += self.get_current_state_for_stage_layer(
                    "postprocessing", "fwd", True, False, False
                )

        elif stage_name == "postprocessing":
            if has_preprocesing:
                states += self.get_current_state_for_stage_layer(
                    "preprocessing", "bwd", False, False, False
                )

            states += self.get_current_state_for_stage_layer(
                "block", "bwd", False, False, False
            ) * (n_block_layers - 1)
            states += self.get_current_state_for_stage_layer(
                "block", "fwd", True, False, False
            )
            states += self.get_current_state_for_stage_layer(
                "postprocessing", "fwd", True, False, False
            )

        return states

    def get_current_states_for_model_bwd(self, stage_name, block_idx=0):
        states = 0
        n_block_layers = self.analyzer.stage_config["block"].get("n_layers")

        has_preprocesing = "preprocessing" in self.analyzer.stages
        has_postprocessing = "postprocessing" in self.analyzer.stages

        if stage_name == "postprocessing":
            states += self.get_current_state_for_stage_layer(
                "postprocessing", "bwd", True, True, False
            )
            states += self.get_current_state_for_stage_layer(
                "block", "bwd", True, True, False
            )
            states += self.get_current_state_for_stage_layer(
                "block", "bwd", False, False, False
            ) * (n_block_layers - 1)
            if has_preprocesing:
                states += self.get_current_state_for_stage_layer(
                    "preprocessing", "bwd", False, False, False
                )

        elif stage_name == "block":
            if has_postprocessing:
                states += self.get_current_state_for_stage_layer(
                    "postprocessing", "fwd", False, False, False
                )

            if block_idx >= 1:
                states += self.get_current_state_for_stage_layer(
                    "block", "fwd", False, False, False
                ) * (n_block_layers - block_idx - 1)
                states += (
                    self.get_current_state_for_stage_layer(
                        "block", "bwd", True, True, False
                    )
                    * 2
                )
                states += self.get_current_state_for_stage_layer(
                    "block", "bwd", False, False, False
                ) * (block_idx - 1)
                if has_preprocesing:
                    states += self.get_current_state_for_stage_layer(
                        "preprocessing", "bwd", False, False, False
                    )
            elif not has_preprocesing:
                states += self.get_current_state_for_stage_layer(
                    "block", "fwd", False, False, False
                ) * (n_block_layers - 2)
                states += self.get_current_state_for_stage_layer(
                    "block", "fwd", True, False, False
                )
                states += self.get_current_state_for_stage_layer(
                    "block", "bwd", True, True, False
                )
            else:
                states += self.get_current_state_for_stage_layer(
                    "block", "fwd", False, False, False
                ) * (n_block_layers - 1)
                states += self.get_current_state_for_stage_layer(
                    "block", "fwd", True, True, False
                )
                states += self.get_current_state_for_stage_layer(
                    "preprocessing", "bwd", True, True, False
                )

        elif stage_name == "preprocessing":
            if has_postprocessing:
                states += self.get_current_state_for_stage_layer(
                    "postprocessing", "fwd", False, False, False
                )
            states += self.get_current_state_for_stage_layer(
                "block", "fwd", False, False, False
            ) * (n_block_layers - 1)
            states += self.get_current_state_for_stage_layer(
                "block", "bwd", True, True, False
            )
            states += self.get_current_state_for_stage_layer(
                "preprocessing", "bwd", True, True, False
            )

        return states

    def forward_summary(self):
        peak = OrderedDict()
        saved = 0

        has_preprocesing = "preprocessing" in self.analyzer.stages
        has_postprocessing = "postprocessing" in self.analyzer.stages

        if has_preprocesing:
            preprocessing_info = self.stage_info["preprocessing"]
            _peak_ckpt_intermediate = preprocessing_info["ckpt_peak_inside_stage_fwd"]
            _peak_fwd_intermediate = preprocessing_info["fwd_peak_inside_stage_fwd"]
            _saved_tensors_ckpt = preprocessing_info["ckpt_saved_tensors"]
            _saved_tensors_fwd = preprocessing_info["fwd_saved_tensors"]
            _saved_tensors_ckpt_memory = compute_memory_for_flattened(
                _saved_tensors_ckpt
            )
            _saved_tensors_fwd_memory = compute_memory_for_flattened(_saved_tensors_fwd)
            _states = self.get_current_states_for_model_fwd("preprocessing")
            peak["preprocessing"] = {
                "ckpt_intermediate": _peak_ckpt_intermediate,
                "fwd_intermediate": _peak_fwd_intermediate,
                "states": _states,
                "prev_saved": 0,
            }
            _preprocess_ckpt_num = self.analyzer.stage_config["preprocessing"].get(
                "ckpt_block_num"
            )
            _peak = (
                sp.Piecewise(
                    (_peak_ckpt_intermediate, sp.Eq(_preprocess_ckpt_num, 1)),
                    (_peak_fwd_intermediate, sp.Eq(_preprocess_ckpt_num, 0)),
                    (-1e9, True),
                )
                + _states
            )
            saved += sp.Piecewise(
                (_saved_tensors_ckpt_memory, sp.Eq(_preprocess_ckpt_num, 1)),
                (_saved_tensors_fwd_memory, sp.Eq(_preprocess_ckpt_num, 0)),
                (-1e9, True),
            )
            peak["preprocessing"].update(peak=_peak, after_saved=saved)

        # Blocks
        block_info = self.stage_info["block"]
        block_config = self.analyzer.stage_config["block"]
        n_layers = block_config.get("n_layers")
        ckpt_block_num = block_config.get("ckpt_block_num")

        _peak_ckpt_intermediate = block_info["ckpt_peak_inside_stage_fwd"]
        _peak_fwd_intermediate = block_info["fwd_peak_inside_stage_fwd"]
        _saved_tensors_ckpt = block_info["ckpt_saved_tensors"]
        _saved_tensors_fwd = block_info["fwd_saved_tensors"]
        _saved_tensors_fwd_memory = compute_memory_for_flattened(_saved_tensors_fwd)
        _saved_tensors_ckpt_memory = compute_memory_for_flattened(_saved_tensors_ckpt)

        assert isinstance(
            n_layers, numbers.Integral
        ), f"n_layers must be an integer, got {n_layers}"

        for i in range(n_layers):
            _states = self.get_current_states_for_model_fwd("block", i)
            peak[f"block_{i}"] = {
                "ckpt_intermediate": _peak_ckpt_intermediate,
                "fwd_intermediate": _peak_fwd_intermediate,
                "states": _states,
                "prev_saved": saved,
            }
            checkpointed = i < ckpt_block_num
            _peak = (
                sp.Piecewise(
                    (_peak_ckpt_intermediate, checkpointed),
                    (_peak_fwd_intermediate, True),
                )
                + _states
                + saved
            )
            saved += sp.Piecewise(
                (_saved_tensors_ckpt_memory, checkpointed),
                (_saved_tensors_fwd_memory, True),
            )
            peak[f"block_{i}"].update(peak=_peak, after_saved=saved)

        if has_postprocessing:
            postprocessing_info = self.stage_info["postprocessing"]
            _peak_fwd_intermediate = postprocessing_info["fwd_peak_inside_stage_fwd"]
            _saved_tensors_fwd = postprocessing_info["fwd_saved_tensors"]
            _states = self.get_current_states_for_model_fwd("postprocessing")
            peak["postprocessing"] = {
                "peak": _peak_fwd_intermediate + _states + saved,
                "fwd_intermediate": _peak_fwd_intermediate,
                "states": _states,
                "prev_saved": saved,
            }
            saved += compute_memory_for_flattened(_saved_tensors_fwd)
            peak["postprocessing"].update(after_saved=saved)

        self.fwd_peaks = peak
        self.fwd_peak = sp.Max(*[v["peak"] for v in peak.values()])

        return peak

    def backward_summary(self):
        peak = OrderedDict()

        has_preprocesing = "preprocessing" in self.analyzer.stages
        has_postprocessing = "postprocessing" in self.analyzer.stages

        if has_postprocessing:
            postprocessing_info = self.stage_info["postprocessing"]
            _peak_bwd_intermediate = postprocessing_info["bwd_peak_inside_stage_bwd"]
            _states = self.get_current_states_for_model_bwd("postprocessing")
            _saved = self.fwd_peaks["postprocessing"]["prev_saved"]
            peak["postprocessing"] = {
                "peak": _peak_bwd_intermediate + _states + _saved,
                "bwd_intermediate": _peak_bwd_intermediate,
                "states": _states,
            }

        # Blocks
        block_info = self.stage_info["block"]
        block_config = self.analyzer.stage_config["block"]
        n_layers = block_config.get("n_layers")

        _peak_bwd_intermediate = block_info["bwd_peak_inside_stage_bwd"]

        assert isinstance(
            n_layers, numbers.Integral
        ), f"n_layers must be an integer, got {n_layers}"

        for i in reversed(range(n_layers)):
            _states = self.get_current_states_for_model_bwd("block", i)
            _saved = self.fwd_peaks[f"block_{i}"]["prev_saved"]
            peak[f"block_{i}"] = {
                "peak": _peak_bwd_intermediate + _states + _saved,
                "bwd_intermediate": _peak_bwd_intermediate,
                "states": _states,
            }

        if has_preprocesing:
            preprocessing_info = self.stage_info["preprocessing"]
            _peak_bwd_intermediate = preprocessing_info["bwd_peak_inside_stage_bwd"]
            _states = self.get_current_states_for_model_bwd("preprocessing")
            _saved = self.fwd_peaks["preprocessing"]["prev_saved"]
            peak["preprocessing"] = {
                "peak": _peak_bwd_intermediate + _states + _saved,
                "bwd_intermediate": _peak_bwd_intermediate,
                "states": _states,
            }

        self.bwd_peaks = peak
        self.bwd_peak = sp.Max(*[v["peak"] for v in peak.values()])

        return peak

    def dump(self, path="./", filename="memory.log"):
        file_path = os.path.join(path, filename)
        output_dict = {
            "fwd_peaks": dict(self.fwd_peaks),
            "bwd_peaks": dict(self.bwd_peaks),
            "stage_info": dict(self.stage_info),
            "fwd_peak": self.fwd_peak,
            "bwd_peak": self.bwd_peak,
        }
        pprint_to_file(
            output_dict, mode="w", filename=file_path, to_screen=False, width=120
        )


class ThroughputSummarizer:
    def __init__(self, analyzer: MistSymbolicAnalyzer):
        self.analyzer = analyzer
        self.stage_info = OrderedDict({stage: {} for stage in self.analyzer.stages})

        for stage in self.analyzer.stages:
            self.stage_summary(stage)

    def stage_summary(self, stage_name: str):
        analyzer = self.analyzer.stage_analyzer[stage_name]
        config = analyzer.config
        module = analyzer.module
        nodes = [n for n in module.graph.nodes]
        symbol2variable = config.symbol2variable()

        # Computations
        node2computation = OrderedDict()
        all_fwd = 0
        all_bwd = 0
        for node in nodes:
            if node.op in {"placeholder", "output", "get_attr"}:
                continue
            info = analyzer.fwd_node2info[node]

            if node.target in {"size", "view", "to", "type", "contiguous"}:
                continue

            if getattr(node.target, "__name__", None) in {
                "getitem",
                "getattr",
                "full",
                "finfo",
            }:
                continue

            symbolic_node_spec = SymbolicNodeSpec.from_fx_node(
                node, *info.concrete_args, **info.concrete_kwargs
            )

            # Get all symbols and their choices, and combine them
            all_choices = []
            for symbol in symbolic_node_spec.symbols():
                if symbol not in symbol2variable:
                    raise RuntimeError(f"Symbol {symbol} not found in symbol2variable")
                all_choices.append(symbol2variable[symbol].choices)
            all_choices = list(product(*all_choices))

            # Not all combinations of choices are valid, filter out the invalid ones
            all_valid_mapping = []
            for choices in all_choices:
                mapping = dict(zip(symbolic_node_spec.symbols(), choices))
                if config.satisfy_constraints(mapping):
                    all_valid_mapping.append(mapping)

            # Profile the concrete node spec for each valid mapping and
            # get the piecewise computation function for the node
            fwd_computation = []
            bwd_computation = []
            for mapping in all_valid_mapping:
                concrete_node_spec = symbolic_node_spec.concretize(mapping)
                (fwd_latency, _, _), (bwd_latency, _, _) = concrete_node_spec.profile()
                fwd_computation.append(
                    (fwd_latency, sp.And(*[sp.Eq(k, v) for k, v in mapping.items()])),
                )
                bwd_computation.append(
                    (bwd_latency, sp.And(*[sp.Eq(k, v) for k, v in mapping.items()])),
                )
            fwd_computation.append(
                (-1e9, True),
            )
            bwd_computation.append(
                (-1e9, True),
            )
            fwd_computation = sp.Piecewise(*fwd_computation)
            bwd_computation = sp.Piecewise(*bwd_computation)

            all_fwd += fwd_computation
            all_bwd += bwd_computation

            # Record the computation
            node2computation[node.name] = {
                # "node": node,
                # "symbolic_node_spec": symbolic_node_spec,
                # "all_valid_mapping": all_valid_mapping,
                "node_spec_str": str(symbolic_node_spec),
                "fwd_computation": fwd_computation,
                "bwd_computation": bwd_computation,
            }

        # Communication
        config = self.analyzer.stage_config[stage_name]
        wgt_fwd_rdn_sharding_size = config.get("wgt_fwd_rdn_sharding_size")
        wgt_bwd_rdn_sharding_size = config.get("wgt_bwd_rdn_sharding_size")
        wgt_fwd_ofd_ratio = config.get("wgt_fwd_ofd_ratio")
        wgt_bwd_ofd_ratio = config.get("wgt_bwd_ofd_ratio")
        grad_ofd_ratio = config.get("grad_ofd_ratio")  # No sharding for grad
        opt_rdn_sharding_size = config.get("opt_rdn_sharding_size")
        opt_ofd_ratio = config.get("opt_ofd_ratio")

        # GPU-CPU communication
        module = self.analyzer.stage_module[stage_name]
        params, buffers = get_params_and_buffers(module)
        full_weights = params + buffers
        # Weights
        sharded_weights_fwd = full_weights / wgt_fwd_rdn_sharding_size
        sharded_and_offloaded_weights_in_gpu_fwd = sharded_weights_fwd * (
            1 - wgt_fwd_ofd_ratio
        )
        sharded_weights_bwd = full_weights / wgt_bwd_rdn_sharding_size
        sharded_and_offloaded_weights_in_gpu_bwd = sharded_weights_bwd * (
            1 - wgt_bwd_ofd_ratio
        )
        # Grads
        full_grads = full_weights
        offloaded_grads_in_gpu = full_grads * (1 - grad_ofd_ratio)
        # Volume
        fwd_gpu_cpu_comm_volume = (
            sharded_weights_fwd - sharded_and_offloaded_weights_in_gpu_fwd
        )
        bwd_gpu_cpu_comm_volume = (
            sharded_weights_bwd - sharded_and_offloaded_weights_in_gpu_bwd
        ) + (full_grads - offloaded_grads_in_gpu)

        # GPU-GPU communication
        fwd_allgather_volume = full_weights - sharded_weights_fwd
        bwd_allgather_volume = full_weights - sharded_weights_bwd

        # BW
        gpu_cpu_bandwidth = config.get("gpu_cpu_bandwidth")
        gpu_gpu_bandwidth = config.get("gpu_gpu_bandwidth")

        self.stage_info[stage_name] = {
            "node2computation": node2computation,
            "all_fwd": all_fwd,
            "all_bwd": all_bwd,
            "fwd_gpu_cpu_comm_volume": fwd_gpu_cpu_comm_volume,
            "bwd_gpu_cpu_comm_volume": bwd_gpu_cpu_comm_volume,
            "fwd_gpu_cpu_comm_latency": fwd_gpu_cpu_comm_volume / gpu_cpu_bandwidth,
            "bwd_gpu_cpu_comm_latency": bwd_gpu_cpu_comm_volume / gpu_cpu_bandwidth,
            "fwd_allgather_volume": fwd_allgather_volume,
            "bwd_allgather_volume": bwd_allgather_volume,
            "fwd_allgather_latency": fwd_allgather_volume / gpu_gpu_bandwidth,
            "bwd_allgather_latency": bwd_allgather_volume / gpu_gpu_bandwidth,
        }
        self.stage_info[stage_name] = tree_map(sp2py, self.stage_info[stage_name])

    def summary(self):
        has_preprocessing = "preprocessing" in self.analyzer.stages
        has_postprocessing = "postprocessing" in self.analyzer.stages

        fwd_time = 0
        bwd_time = 0

        if has_preprocessing:
            fwd_computation = self.stage_info["preprocessing"]["all_fwd"]
            bwd_computation = self.stage_info["preprocessing"]["all_bwd"]
            pre_ckpt_num = self.analyzer.stage_config["preprocessing"].get(
                "ckpt_block_num"
            )
            ckpt_computation = fwd_computation * pre_ckpt_num

            fwd_time += fwd_computation
            bwd_time += ckpt_computation + bwd_computation

        # Blocks
        block_info = self.stage_info["block"]
        block_config = self.analyzer.stage_config["block"]
        n_layers = block_config.get("n_layers")
        ckpt_block_num = block_config.get("ckpt_block_num")

        fwd_computation = block_info["all_fwd"]
        bwd_computation = block_info["all_bwd"]

        fwd_gpu_cpu_comm_latency = block_info["fwd_gpu_cpu_comm_latency"]
        bwd_gpu_cpu_comm_latency = block_info["bwd_gpu_cpu_comm_latency"]
        fwd_allgather_latency = block_info["fwd_allgather_latency"]
        bwd_allgather_latency = block_info["bwd_allgather_latency"]

        fwd_time_for_one_layer = sp.Max(
            fwd_computation, fwd_gpu_cpu_comm_latency + fwd_allgather_latency
        )
        bwd_time_for_one_layer_with_ckpt = sp.Max(
            fwd_computation + bwd_computation,
            bwd_gpu_cpu_comm_latency + bwd_allgather_latency,
        )
        bwd_time_for_one_layer_without_ckpt = sp.Max(
            bwd_computation,
            bwd_gpu_cpu_comm_latency + bwd_allgather_latency,
        )

        fwd_time += fwd_time_for_one_layer * (n_layers - 1)
        fwd_time += fwd_computation

        bwd_time += sp.Piecewise(
            (
                bwd_time_for_one_layer_with_ckpt * (n_layers - 1)
                + fwd_computation
                + bwd_computation,
                sp.Eq(ckpt_block_num, n_layers),
            ),
            (
                bwd_time_for_one_layer_without_ckpt * (n_layers - 1) + bwd_computation,
                sp.Eq(ckpt_block_num, 0),
            ),
            (
                bwd_time_for_one_layer_without_ckpt * (n_layers - ckpt_block_num)
                + bwd_time_for_one_layer_with_ckpt * (ckpt_block_num - 1)
                + fwd_computation
                + bwd_computation,
                True,
            ),
        )

        # Postprocessing
        if has_postprocessing:
            fwd_computation = self.stage_info["postprocessing"]["all_fwd"]
            bwd_computation = self.stage_info["postprocessing"]["all_bwd"]

            fwd_time += fwd_computation
            bwd_time += bwd_computation

        self.info = {
            "fwd_time": fwd_time,
            "bwd_time": bwd_time,
        }
        self.info = tree_map(sp2py, self.info)

        return fwd_computation + bwd_computation

    def dump(self, path="./", filename="throughput.json"):
        file_path = os.path.join(path, filename)
        save_json(self.info, file_path)
        save_json(self.stage_info, file_path, mode="a")
