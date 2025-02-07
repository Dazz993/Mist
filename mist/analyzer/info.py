from __future__ import annotations
from dataclasses import dataclass, field, replace
from functools import partial
from typing import (
    Dict,
    OrderedDict,
    List,
    Set,
    Tuple,
    Union,
    Any,
    Callable,
    TYPE_CHECKING,
)
from pprint import pprint, pformat

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.fx
from sympy.utilities.autowrap import ufuncify, autowrap
from torch.fx import Node
from torch.utils._pytree import tree_flatten, tree_map

from mist import global_symbol_manager as gsm
from mist.config import MistConfig
from mist.analyzer.recorder import ExecType, ExecInfo, ExecInfoRecorder
from mist.memory_pool import (
    MemoryPool,
    compute_memory_for_set,
    saved_tensors_manager as stm,
)
from mist.logger import get_logger
from mist.utils.tensor_entry import tree_to_entries, get_tensor_base_id
from mist.node_database.infer_grad_inputs import infer_grad_inputs_for_symop
from mist.utils.pytree import tree_flatten_like, tree_zip_map
from mist.utils.tensor_entry import TensorEntry
from mist.utils.sympy import indicator
from mist.node_database.database import NodeDB
from mist.node_database.node_spec import NodeSpec
from mist.node_database.symbolic_node_spec import SymbolicNodeSpec

if TYPE_CHECKING:
    from mist.analyzer.strategy import LayerStrategy, PhaseStrategy

Symbol = sp.Symbol
VarInt = Union[int, Symbol]
VarBool = Union[bool, Symbol]
VarFloat = Union[float, Symbol]

# @dataclass
# class LayerInfo:
#     strategy: LayerStrategy = None
#     exec_type: ExecType = None
#     ckpt: bool = None
#     # Latency
#     symbolic_node_specs: List[SymbolicNodeSpec] = field(default_factory=list)
#     # Memory
#     peak_memory: sp.Expr = None
#     peak_tensor_sets: List[Set[TensorEntry]] = field(default_factory=list)
#     saved_memory: sp.Expr = None
#     saved_tensor_set: Set[TensorEntry] = None
#     # Model Params and Buffers
#     params_require_grad: Set[TensorEntry] = field(default_factory=list)
#     params_not_require_grad: Set[TensorEntry] = field(default_factory=list)
#     buffers: Set[TensorEntry] = field(default_factory=list)
#     # Aux
#     # * weights
#     full_weights: sp.Expr = None
#     sharded_and_offloaded_weights_in_gpu: sp.Expr = None
#     # * grads
#     full_grads: sp.Expr = None
#     sharded_and_offloaded_grads_in_gpu: sp.Expr = None
#     # * opts
#     full_opts: sp.Expr = None
#     sharded_and_offloaded_opts_in_gpu: sp.Expr = None
#     # * flags
#     aux_set: bool = False

#     def calculate_aux(self):
#         if self.aux_set:
#             return
#         # * saved_memory
#         self.saved_memory = compute_memory_for_set(self.saved_tensor_set)
#         # * full
#         self.full_weights = compute_memory_for_set(
#             self.params_require_grad | self.params_not_require_grad | self.buffers
#         )
#         self.full_grads = compute_memory_for_set(self.params_require_grad)
#         self.full_opts = compute_memory_for_set(self.params_require_grad)
#         # * sharded and offloaded
#         strategy = self.strategy
#         self.sharded_and_offloaded_weights_in_gpu = (
#             (1 - strategy.wo_ratio) * self.full_weights / strategy.ws_size
#         )
#         self.sharded_and_offloaded_grads_in_gpu = (
#             (1 - strategy.go_ratio) * self.full_grads / strategy.gs_size
#         )
#         self.sharded_and_offloaded_opts_in_gpu = (
#             (1 - strategy.oo_ratio) * self.full_opts / strategy.os_size
#         )
#         self.aux_set = True

#     def update(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)


@dataclass
class Info:
    def assert_dataclass_complete(self):
        for name, field in self.__dataclass_fields__.items():
            if not name.startswith("_"):
                assert (
                    getattr(self, name) is not None
                ), f"Field '{name}' is not set for {self}"


@dataclass
class LayerInfo(Info):
    # Basic
    strategy: LayerStrategy = None
    fwd_info: PhaseInfo = None
    bwd_info: PhaseInfo = None

    # Auxillary information
    # * Raw memory of params and buffers
    params_require_grad: Set[TensorEntry] = None
    params_not_require_grad: Set[TensorEntry] = None
    buffers: Set[TensorEntry] = None

    def post_process(self, config: MistConfig):
        self.fwd_info.post_process(config)
        self.bwd_info.post_process(config)

    def _identity(self):
        return (
            hash(self.fwd_info),
            hash(self.bwd_info),
        )

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, LayerInfo):
            return False
        return self._identity() == __value._identity()


@dataclass
class PhaseInfo(Info):
    # Parent LayerInfo
    layer_info: LayerInfo = None

    # Strategy used
    strategy: PhaseStrategy = None

    # Necessary information for the phase
    # * Activation Memory
    saved: VarInt = None
    peak: VarInt = None
    output: VarInt = None
    # * States Memory
    full_weights: VarInt = None
    partial_weights: VarInt = None
    full_grads: VarInt = None
    partial_grads: VarInt = None
    full_opts: VarInt = None
    partial_opts: VarInt = None

    # Auxillary information
    _peak_tensor_sets: List[Set[TensorEntry]] = field(default_factory=list)
    _saved_tensor_set: Set[TensorEntry] = None
    _symbolic_node_specs: List[SymbolicNodeSpec] = field(default_factory=list)
    _eval_funcs: Dict[str, Callable] = field(default_factory=dict)

    def post_process(self, config: MistConfig):
        params_require_grad = self.layer_info.params_require_grad
        params_not_require_grad = self.layer_info.params_not_require_grad
        buffers = self.layer_info.buffers

        full_weights = compute_memory_for_set(
            params_require_grad | params_not_require_grad | buffers
        )
        full_grads = compute_memory_for_set(params_require_grad)
        full_opts = compute_memory_for_set(params_require_grad)

        # TODO(zhanda): Check dtype information
        self.full_weights = gsm.subs(full_weights, self.strategy.mapping)
        self.full_grads = gsm.subs(full_grads, self.strategy.mapping)
        full_opts = gsm.subs(full_opts, self.strategy.mapping)
        # If it is trained with half precision while optimizer is in full precision,
        # then we would keep extra fp32 weights in GPU. At that time, the full_opts
        # would be 4 x (fp16 weights) for Adam and 2 x (fp16 weights) for Fp32 weights.
        training_config = config.training
        # training_config = config
        if (
            training_config.params_dtype in [torch.float16, torch.bfloat16]
            and training_config.optimizer_dtype == torch.float32
        ):
            if training_config.optimizer_name in ["adam", "adamw"]:
                self.full_opts = full_opts * 6
            elif training_config.optimizer_name == "sgd":
                self.full_opts = full_opts * 4
            else:
                raise ValueError(
                    f"Unsupported optimizer name: {training_config.optimizer_name}"
                )
        else:
            self.full_opts = full_opts * 2

        # * sharded and offloaded
        strategy = self.strategy
        self.partial_weights = (
            (1 - strategy.wo_ratio) * self.full_weights / strategy.ws_size
        )
        self.partial_grads = (
            (1 - strategy.go_ratio) * self.full_grads / strategy.gs_size
        )
        self.partial_opts = (1 - strategy.oo_ratio) * self.full_opts / strategy.os_size

    def get_concrete_memory(self, item: str, symbol_mapping: Dict[str, Any]):
        assert item in self.__dataclass_fields__, f"Invalid item: {item}"
        old_symbol_mapping = self.strategy.generate_name_to_symbol_dict()
        assert all(
            s1 == s2 for s1, s2 in zip(old_symbol_mapping.keys(), symbol_mapping.keys())
        ), (
            f"Symbol mapping keys {old_symbol_mapping.keys()} "
            f"does not match with {symbol_mapping.keys()}"
        )

        if item in self._eval_funcs:
            return self._eval_funcs[item](*symbol_mapping.values())
        else:
            old_symbols = tuple(old_symbol_mapping.values())
            # TODO(zhanda): use this batched ufuncify later
            # func = ufuncify(
            #     old_symbols,
            #     getattr(self, item),
            #     language="C",
            #     backend="cython",
            #     verbose=True,
            # )
            # self._eval_funcs[item] = func
            # return func(*(np.array([v], dtype=float) for v in symbol_mapping.values()))
            func = autowrap(
                getattr(self, item),
                language="C",
                backend="cython",
                args=old_symbols,
            )
            self._eval_funcs[item] = func
            return func(*symbol_mapping.values())

        saved = self.saved.subs(symbol_mapping)
        peak = self.peak.subs(symbol_mapping)
        full_weights = self.full_weights.subs(symbol_mapping)
        partial_weights = self.partial_weights.subs(symbol_mapping)
        full_grads = self.full_grads.subs(symbol_mapping)
        partial_grads = self.partial_grads.subs(symbol_mapping)
        full_opts = self.full_opts.subs(symbol_mapping)
        partial_opts = self.partial_opts.subs(symbol_mapping)
        return {
            "saved": saved,
            "peak": peak,
            "full_weights": full_weights,
            "partial_weights": partial_weights,
            "full_grads": full_grads,
            "partial_grads": partial_grads,
            "full_opts": full_opts,
            "partial_opts": partial_opts,
        }

    def _identity(self):
        return (
            self.saved,
            self.peak,
            self.full_weights,
            self.partial_weights,
            self.full_grads,
            self.partial_grads,
            self.full_opts,
            self.partial_opts,
        )

    def __hash__(self) -> int:
        return hash(self._identity())

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PhaseInfo):
            return False
        return self._identity() == __value._identity()
