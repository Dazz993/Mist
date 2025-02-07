from __future__ import annotations
import inspect
from dataclasses import dataclass, field, replace
from functools import partial
from numbers import Number
from typing import Dict, OrderedDict, List, Set, Tuple, Union, Any, TYPE_CHECKING
from pprint import pprint, pformat

import sympy as sp
import torch
import torch.nn as nn
import torch.fx
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
from mist.distributed.overrides import MistProcessGroup
from mist.analyzer.strategy import LayerStrategy, create_strategy_for_a_layer
from mist.analyzer.info import LayerInfo, PhaseInfo
from mist.symbols import temporarily_set_sp_eq_ne
from mist.logger import get_logger
from mist.utils.tensor_entry import tree_to_entries, get_tensor_base_id
from mist.node_database.infer_grad_inputs import infer_grad_inputs_for_symop
from mist.utils.pytree import tree_flatten_like, tree_zip_map
from mist.utils.tensor_entry import TensorEntry
from mist.utils.sympy import indicator
from mist.node_database.database import NodeDB
from mist.node_database.node_spec import NodeSpec
from mist.node_database.symbolic_node_spec import SymbolicNodeSpec
from mist.utils.common import save_json, pprint_to_file
from mist.utils.module import getattr_recursive

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mist.analyzer.strategy import LayerStrategy, PhaseStrategy

Symbol = sp.Symbol
VarInt = Union[int, Symbol]
VarBool = Union[bool, Symbol]
VarFloat = Union[float, Symbol]

BATCH_SIZE_CANDIDATES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
TP_SIZE_CANDIDATES = tuple(range(1, 1025))


def print_dict(d, printer, prefix=""):
    d = dict(d)
    for k, v in d.items():
        printer(f"{prefix}{k}: {v}")


def print_seq(seq, printer, prefix=""):
    if not isinstance(seq, (list, tuple)):
        seq = [seq]
    for i, item in enumerate(seq):
        printer(f"{prefix}{item}")


def has_torch_tensor(obj):
    return any(isinstance(item, torch.Tensor) for item in tree_flatten(obj)[0])


def exclude_tensors_using_base_id(input_tensors, exclude_tensors):
    """
    Exclude the tensors that have the same base_id as the tensors in the exclude_tensors.
    """
    if exclude_tensors is None:
        return input_tensors

    exclude_base_ids = set()
    for tensor in exclude_tensors:
        base_id = id(tensor) if tensor._base is None else id(tensor._base)
        exclude_base_ids.add(base_id)

    output_tensors = []
    for tensor in input_tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        base_id = id(tensor) if tensor._base is None else id(tensor._base)
        if base_id not in exclude_base_ids:
            output_tensors.append(tensor)

    return output_tensors


def has_non_negligible_latency(node: Node):
    if node.op in {"placeholder", "output", "get_attr"}:
        return False
    if node.op == "call_method" and node.target in {
        "size",
        "view",
        "to",
        "type",
        "contiguous",
        "reshape",
        "permute",
        # "chunk",
        # "split",
    }:
        return False
    if node.op == "call_function" and getattr(node.target, "__name__", None) in {
        "getitem",
        "getattr",
        "finfo",
        "arange",
        "zeros",
        "empty",
        "full",
        "max",
        "min",
    }:
        return False

    if node.target == "split":
        print(node.format_node())
        exit()

    return True


def has_cuda_tensor(obj):
    return any(
        isinstance(item, torch.Tensor) and item.is_cuda for item in tree_flatten(obj)[0]
    )


def is_cuda_kernel(node: Node, outputs, *args, **kwargs):
    has_cuda_tensor_args = has_cuda_tensor(args)
    has_cuda_tensor_kwargs = has_cuda_tensor(kwargs)
    has_cuda_tensor_outputs = has_cuda_tensor(outputs)
    return has_cuda_tensor_args or has_cuda_tensor_kwargs or has_cuda_tensor_outputs


def tree_symbol_subs(tree, mapping):
    def fn(obj):
        if isinstance(obj, set):
            return set(fn(item) for item in obj)
        elif isinstance(obj, TensorEntry):
            out = obj.copy()
            out.shape = gsm.subs(out.shape, mapping)
            return out
        elif isinstance(obj, sp.Basic):
            return gsm.subs(obj, mapping)
        elif isinstance(obj, MistProcessGroup):
            return MistProcessGroup(
                rank=gsm.subs(obj._rank, mapping),
                global_rank=gsm.subs(obj._global_rank, mapping),
                world_size=gsm.subs(obj._world_size, mapping),
            )
        else:
            return obj

    return tree_map(fn, tree)


def map_to_tensor_entry(tree):
    def fn(obj):
        if isinstance(obj, torch.Tensor):
            return TensorEntry.from_tensor(obj)
        return obj

    return tree_map(fn, tree)


def tree_add_shape_constraints(tree, constraints=None):
    constraints = constraints or set()

    def fn(obj):
        if isinstance(obj, (TensorEntry, torch.Tensor)):
            for s in obj.shape:
                if isinstance(s, sp.Basic) and not isinstance(s, Number):
                    constraints.add(s)

    tree_map(fn, tree)
    return constraints


class LayerAnalyzer:
    def __init__(
        self,
        recorder: ExecInfoRecorder,
        layer_strategy: LayerStrategy,
        config: MistConfig,
    ):
        # Get attrs from the recorder
        self.recorder: ExecInfoRecorder = recorder
        self.config: MistConfig = config
        self.module = recorder.module
        self.user_to_last_uses: Dict[Node, List[Node]] = recorder.user_to_last_uses
        self.name2node: Dict[str, None] = recorder.name2node
        self.placeholder_nodes: List[Node] = recorder.placeholder_nodes
        self.output_node: Node = recorder.output_node
        self.node_exec_seq_fwd: List[Node] = recorder.node_exec_seq_fwd
        self.node_exec_seq_bwd: List[Node] = recorder.node_exec_seq_bwd
        self.fwd_node2info: Dict[Node, ExecInfo] = recorder.fwd_node2info
        self.bwd_node2info: Dict[Node, ExecInfo] = recorder.bwd_node2info
        self.exec_type_to_node_to_info: Dict[
            ExecType, Dict[Node, ExecInfo]
        ] = recorder.exec_type_to_node_to_info

        # Set analyzer attributes
        self.layer_strategy: LayerStrategy = layer_strategy
        self.ckpt = layer_strategy.ckpt
        self.fwd_strategy: PhaseStrategy = layer_strategy.fwd_strategy
        self.bwd_strategy: PhaseStrategy = layer_strategy.bwd_strategy

        # Init the info
        self.fwd_info: PhaseInfo = PhaseInfo(strategy=self.fwd_strategy)
        self.bwd_info: PhaseInfo = PhaseInfo(strategy=self.bwd_strategy)
        self.layer_info = LayerInfo(
            strategy=self.layer_strategy,
            fwd_info=self.fwd_info,
            bwd_info=self.bwd_info,
        )
        self.fwd_info.layer_info = self.layer_info
        self.bwd_info.layer_info = self.layer_info

        # Prepare for the analysis
        self.memory_pool: MemoryPool = MemoryPool()
        self.prepare_states_info()

    def prepare_states_info(self):
        params_require_grad: List[torch.Tensor] = [
            p for p in self.module.parameters() if p.requires_grad
        ]
        params_not_require_grad: List[torch.Tensor] = [
            p for p in self.module.parameters() if not p.requires_grad
        ]
        buffers: List[torch.Tensor] = list(self.module.buffers())
        self.exclude_module_attr = partial(
            exclude_tensors_using_base_id,
            exclude_tensors=params_require_grad + params_not_require_grad + buffers,
        )

        self.layer_info.params_require_grad = set(
            TensorEntry.from_tensor(tensor) for tensor in params_require_grad
        )
        self.layer_info.params_not_require_grad = set(
            TensorEntry.from_tensor(tensor) for tensor in params_not_require_grad
        )
        self.layer_info.buffers = set(
            TensorEntry.from_tensor(tensor) for tensor in buffers
        )

    def run(self):
        self.run_forward()
        self.run_backward()
        self.summary()
        return self.layer_info

    def run_forward(self):
        memory_pool = self.memory_pool

        for node in self.node_exec_seq_fwd:
            fwd_info = self.fwd_node2info[node]
            bwd_info = self.bwd_node2info.get(node, None)

            # Get the output tensor
            output = fwd_info.output

            # Save the tensors for the backward pass if needed
            saved_tensors = fwd_info.saved_tensors

            if saved_tensors is not None:
                # Exclude the weights and buffers
                saved_intermediates = self.exclude_module_attr(saved_tensors)
                # Save the tensors in the memory pool
                memory_pool.batch_add(
                    *saved_intermediates,
                    category="saved_tensors",
                    comment=f"saved_for_{node.name}",
                )

            # Save the output only in the outer block
            intermediates = self.exclude_module_attr(tree_flatten(output)[0])
            memory_pool.batch_add(
                *intermediates, category="intermediate", comment=node.name
            )

            # Release the tensors that are not needed anymore
            memory_pool_before_release = memory_pool.copy()
            for to_delete_node in self.user_to_last_uses.get(node, []):
                to_delete = self.fwd_node2info[to_delete_node].output
                to_delete = self.exclude_module_attr(tree_flatten(to_delete)[0])
                memory_pool.batch_remove(*to_delete, category="intermediate")

            # ======================================================================
            # Deprecated: because we assume the output will be used in the outer scope
            # since peak memory is mostly the last block.
            # # NOTE(zhanda-2023/11/27): the output memory is individually considered
            # in the memory analyzer.
            # If the node is the output node, release the output
            if node.op == "output":
                memory_pool.batch_remove(*intermediates, category="intermediate")
            # ======================================================================

            # Update the execution info
            self.fwd_node2info[node].update(
                memory_pool_before_release=memory_pool_before_release,
                memory_pool_after_release=memory_pool.copy(),
            )

            self.log_info(fwd_info, "MemAnalyzer.FWD")

    def run_backward(self):
        memory_pool = self.memory_pool

        # Add the grad of the output node to the memory pool
        output_grad = self.bwd_node2info[self.output_node].output
        memory_pool.batch_add(
            *tree_flatten(output_grad)[0],
            category="intermediate",
            comment=f"grad_for_{self.output_node.name}",
        )

        for node in self.node_exec_seq_bwd:
            fwd_info = self.fwd_node2info[node]
            bwd_info = self.bwd_node2info.get(node, None)

            if bwd_info is None or bwd_info.output is None:
                continue

            flat_arg_nodes, flat_arg_nodes_spec = tree_flatten(node.args)
            flat_kwarg_nodes, flat_kwarg_nodes_spec = tree_flatten(node.kwargs)
            flat_arg_grads, _ = tree_flatten_like(bwd_info.args, flat_arg_nodes_spec)
            flat_kwarg_grads, _ = tree_flatten_like(
                bwd_info.kwargs, flat_kwarg_nodes_spec
            )

            flat_input_nodes = flat_arg_nodes + flat_kwarg_nodes
            flat_input_grads = flat_arg_grads + flat_kwarg_grads

            accum_grads_to_delete = set()
            for input_node, input_grad in zip(flat_input_nodes, flat_input_grads):
                if not isinstance(input_node, Node):
                    continue
                if not has_torch_tensor(input_grad):
                    continue

                # Add intermediate grads to the memory pool
                memory_pool.batch_add(
                    *tree_flatten(input_grad)[0],
                    category="intermediate",
                    comment=f"grad_for_{input_node.name}",
                )

                # Deal with accum_grads
                input_node_grads = self.bwd_node2info[input_node].output
                flat_input_node_grads, _ = tree_flatten(input_node_grads)
                flat_input_grads, _ = tree_flatten(input_grad)
                assert len(flat_input_node_grads) == len(flat_input_grads)
                for real_grads, tmp_grads in zip(
                    flat_input_node_grads, flat_input_grads
                ):
                    if not isinstance(tmp_grads, torch.Tensor):
                        continue
                    if not isinstance(real_grads, torch.Tensor):
                        continue
                    if get_tensor_base_id(real_grads) != get_tensor_base_id(tmp_grads):
                        accum_grads_to_delete.add(tmp_grads)

            # Save the memory pool before release
            memory_pool_before_release = memory_pool.copy()

            # Release the tensors that are not needed anymore
            # 1. Release the tmp accum_grads
            # 2. Release saved_tensors
            # 3. Release output grad of the node
            for input_grad in accum_grads_to_delete:
                memory_pool.batch_remove(
                    *tree_flatten(input_grad)[0], category="intermediate"
                )

            saved_tensors = bwd_info.saved_tensors
            if saved_tensors is not None:
                saved_tensors = self.exclude_module_attr(saved_tensors)
                memory_pool.batch_remove(*saved_tensors, category="saved_tensors")

            memory_pool.batch_remove(
                *tree_flatten(bwd_info.output)[0], category="intermediate"
            )

            # Update the execution info
            self.bwd_node2info[node].update(
                memory_pool_before_release=memory_pool_before_release,
                memory_pool_after_release=memory_pool.copy(),
            )

            self.log_info(bwd_info, "MemAnalyzer.BWD")

    def summary(self):
        # Latency summary
        self.latency_summary()

        # Memory summary
        # * Forward pass
        fwd_peak_with_ckpt, fwd_saved_with_ckpt = self.memory_fwd_summary(ckpt=True)
        fwd_peak_without_ckpt, fwd_saved_without_ckpt = self.memory_fwd_summary(
            ckpt=False
        )
        self.fwd_info.peak = sp.Piecewise(
            (fwd_peak_with_ckpt, sp.Eq(self.ckpt, 1)),
            (fwd_peak_without_ckpt, sp.Eq(self.ckpt, 0)),
            (1e12, True),
        )
        self.fwd_info.saved = sp.Piecewise(
            (fwd_saved_with_ckpt, sp.Eq(self.ckpt, 1)),
            (fwd_saved_without_ckpt, sp.Eq(self.ckpt, 0)),
            (1e12, True),
        )
        # Update the memory for outputs
        outputs = tree_flatten(self.fwd_node2info[self.output_node].output)[0]
        self.fwd_info.output = tree_symbol_subs(
            compute_memory_for_set(outputs), self.fwd_strategy.mapping
        )

        # * Backward pass
        self.bwd_info.peak = self.memory_bwd_summary()

        # Layer Info Post-processing
        self.layer_info.post_process(self.config)

        return self.layer_info

    def latency_summary(self):
        symbolic_node_specs = []
        constraints = set()
        for node in self.node_exec_seq_fwd:
            if node.op in {"placeholder", "output"}:
                continue
            fwd_info = self.fwd_node2info[node]
            args = map_to_tensor_entry(fwd_info.args)
            args = tree_symbol_subs(args, self.fwd_strategy.mapping)
            kwargs = map_to_tensor_entry(fwd_info.kwargs)
            kwargs = tree_symbol_subs(kwargs, self.fwd_strategy.mapping)
            output = map_to_tensor_entry(fwd_info.output)
            output = tree_symbol_subs(output, self.fwd_strategy.mapping)

            if has_non_negligible_latency(node) and is_cuda_kernel(
                node, fwd_info.output, *fwd_info.args, **fwd_info.kwargs
            ):
                symbolic_node_spec = SymbolicNodeSpec.from_fx_node(
                    node, *args, **kwargs
                )
                if node.op == "call_module":
                    symbolic_node_spec.target_spec.constants = tree_symbol_subs(
                        symbolic_node_spec.target_spec.constants,
                        self.fwd_strategy.mapping,
                    )
                logger.debug(f"SymbolicNodeSpec: {symbolic_node_spec}")
                symbolic_node_specs.append(symbolic_node_spec)

            # Add the constraints
            constraints = tree_add_shape_constraints(args, constraints)
            constraints = tree_add_shape_constraints(kwargs, constraints)
            constraints = tree_add_shape_constraints(output, constraints)

        # Add param grad accumulation nodes
        flattened = True
        grad_accumulation_node_specs = []
        params = set(list(self.module.parameters()))
        if flattened:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                num_elements = sum(
                    p.numel() for p in params if p.dtype == dtype and p.requires_grad
                )
                if num_elements == 0:
                    continue
                tensor_entry = TensorEntry(
                    torch.Tensor,
                    shape=(num_elements,),
                    dtype=dtype,
                    device=torch.device(torch.cuda.current_device()),
                    requires_grad=True,
                    id=None,
                    base_id=None,
                )
                tensor_entry = tree_symbol_subs(tensor_entry, self.fwd_strategy.mapping)
                grad_accumulation_node_specs.append(
                    SymbolicNodeSpec.from_callable(
                        torch.add, tensor_entry, tensor_entry
                    )
                )
        else:
            for p in params:
                if p.requires_grad:
                    tensor_entry = TensorEntry.from_tensor(p)
                    tensor_entry = tree_symbol_subs(
                        tensor_entry, self.fwd_strategy.mapping
                    )
                    grad_accumulation_node_specs.append(
                        SymbolicNodeSpec.from_callable(
                            torch.add, tensor_entry, tensor_entry
                        )
                    )

        self.fwd_info._symbolic_node_specs = symbolic_node_specs
        self.bwd_info._symbolic_node_specs = symbolic_node_specs
        self.bwd_info._grad_accumulation_node_specs = grad_accumulation_node_specs
        self.layer_info.constraints = constraints

    def memory_fwd_summary(self, ckpt=False):
        # ckpt should be a concrete value for clear analysis
        assert not isinstance(ckpt, sp.Symbol)

        peak_memory = 0
        peak_tensor_sets = []

        if ckpt:
            saved_tensor_set = set()
            for node in self.placeholder_nodes:
                fwd_info = self.fwd_node2info[node]
                for tensor in tree_flatten(fwd_info.output)[0]:
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    saved_tensor_set.add(TensorEntry.from_tensor(tensor))
        else:
            # Get the after saved tensors
            node = self.output_node
            output_node_memory_pool = self.fwd_node2info[node].memory_pool_after_release
            saved_tensor_set = set(
                output_node_memory_pool.get_category("saved_tensors").keys()
            )

        for node in self.node_exec_seq_fwd:
            fwd_info = self.fwd_node2info[node]
            memory_pool_before_release = fwd_info.memory_pool_before_release

            # Check the peak memory
            intermediate_tensors = set(
                memory_pool_before_release.get_category("intermediate").keys()
            )
            if ckpt:
                saved_tensors = saved_tensor_set
            else:
                saved_tensors = saved_tensors = set(
                    memory_pool_before_release.get_category("saved_tensors").keys()
                )
            cur_peak_tensors = intermediate_tensors | saved_tensors
            cur_peak_memory = compute_memory_for_set(cur_peak_tensors)

            # Consider the inner tensors (e.g. when op is not primitive)
            if fwd_info.extra_inner_for_fwd is not None:
                extra_inner_memory = compute_memory_for_set(
                    fwd_info.extra_inner_for_fwd
                )
                cur_peak_tensors.update(fwd_info.extra_inner_for_fwd)
                cur_peak_memory += extra_inner_memory

            if isinstance(cur_peak_memory - peak_memory, Number):
                if cur_peak_memory - peak_memory >= 0:
                    peak_memory = cur_peak_memory
                    peak_tensor_sets = [cur_peak_tensors]
            elif (cur_peak_memory - peak_memory).is_nonnegative:
                peak_memory = cur_peak_memory
                peak_tensor_sets = [cur_peak_tensors]
            elif (peak_memory - cur_peak_memory).is_positive:
                pass
            else:
                peak_memory = sp.Max(peak_memory, cur_peak_memory)
                peak_tensor_sets.append(cur_peak_tensors)

        peak_memory = tree_symbol_subs(peak_memory, self.fwd_strategy.mapping)
        saved_memory = tree_symbol_subs(
            compute_memory_for_set(saved_tensor_set), self.fwd_strategy.mapping
        )

        # For debugging
        # saved_tensor_set = tree_symbol_subs(saved_tensor_set, self.fwd_strategy.mapping)
        # peak_tensor_sets = tree_symbol_subs(peak_tensor_sets, self.fwd_strategy.mapping)

        return peak_memory, saved_memory

    def memory_bwd_summary(self, ckpt=False):
        """
        For bwd memory analysis, ckpt or not does not matter that much.
        The only difference here is whether to save the grad_outputs of the module.
        """
        peak_memory = 0
        peak_tensor_sets = []

        for node in self.node_exec_seq_bwd:
            bwd_info = self.bwd_node2info.get(node, None)
            if bwd_info is None or bwd_info.output is None:
                continue

            memory_pool_before_release = bwd_info.memory_pool_before_release.copy()

            if ckpt:
                # Add the grad of the output node to the memory pool
                output_grad = self.bwd_node2info[self.output_node].output
                memory_pool_before_release.batch_add(
                    *tree_flatten(output_grad)[0],
                    category="intermediate",
                    comment=f"grad_for_module_ckpt",
                )

            # Check the peak memory
            intermediate_tensors = set(
                memory_pool_before_release.get_category("intermediate").keys()
            )
            saved_tensors = set(
                memory_pool_before_release.get_category("saved_tensors").keys()
            )
            # TODO(zhanda): given different layer strategies for fwd and bwd,
            # the saved tensors in the bwd pass may be different from the fwd pass.
            cur_peak_tensors = intermediate_tensors | saved_tensors
            cur_peak_memory = compute_memory_for_set(cur_peak_tensors)

            # Consider the inner tensors (e.g. when op is not primitive)
            if bwd_info.extra_inner_for_bwd is not None:
                extra_inner_memory = compute_memory_for_set(
                    bwd_info.extra_inner_for_bwd
                )
                cur_peak_tensors.update(bwd_info.extra_inner_for_bwd)
                cur_peak_memory += extra_inner_memory

            if isinstance(cur_peak_memory - peak_memory, Number):
                if cur_peak_memory - peak_memory >= 0:
                    peak_memory = cur_peak_memory
                    peak_tensor_sets = [cur_peak_tensors]
            elif (cur_peak_memory - peak_memory).is_nonnegative:
                peak_memory = cur_peak_memory
                peak_tensor_sets = [cur_peak_tensors]
            elif (peak_memory - cur_peak_memory).is_positive:
                pass
            else:
                peak_memory = sp.Max(peak_memory, cur_peak_memory)
                peak_tensor_sets.append(cur_peak_tensors)

        peak_memory = tree_symbol_subs(peak_memory, self.bwd_strategy.mapping)

        # For debugging
        # bwd_info.peak_tensor_sets = tree_symbol_subs(
        #     peak_tensor_sets, self.bwd_strategy.mapping
        # )

        return peak_memory

    def log_info(self, info, exec_type=None):
        node = info.node
        exec_type = exec_type or info.exec_type
        prefix = "grad_" if exec_type == ExecType.BWD else ""

        logger.debug(
            f"[{exec_type}] Mod ['{self.module.name}'] Node [{node.name}]: {node.format_node()}"
        )

        _print_seq = partial(print_seq, printer=logger.debug, prefix="----- ")
        logger.debug(f"--- {prefix}args: ")
        _print_seq(info.args)
        logger.debug(f"--- {prefix}kwargs: ")
        _print_seq(info.kwargs)
        logger.debug(f"--- {prefix}output: ")
        _print_seq(info.output)

        _print_dict = partial(print_dict, printer=logger.debug, prefix="--------- ")

        pool_before_release = info.memory_pool_before_release
        pool_after_release = info.memory_pool_after_release

        extra_inner_for_fwd = getattr(info, "extra_inner_for_fwd", None)
        extra_inner_for_bwd = getattr(info, "extra_inner_for_bwd", None)
        if extra_inner_for_fwd is not None:
            logger.debug(f"--- {prefix}extra_inner_for_fwd: ")
            _print_seq(extra_inner_for_fwd)
        if extra_inner_for_bwd is not None:
            logger.debug(f"--- {prefix}extra_inner_for_bwd: ")
            _print_seq(extra_inner_for_bwd)

        logger.debug("--- snapshot before release:")
        logger.debug("------ intermediate:")
        _print_dict(pool_before_release.get_category("intermediate"))
        logger.debug("------ saved_tensors:")
        _print_dict(pool_before_release.get_category("saved_tensors"))
        logger.debug("--- snapshot after release:")
        logger.debug("------ intermediate:")
        _print_dict(pool_after_release.get_category("intermediate"))
        logger.debug("------ saved_tensors:")
        _print_dict(pool_after_release.get_category("saved_tensors"))


def analyze_blocks(
    blocks: List[nn.Module],
    inputs: Union[Dict[str, Any], List[Any]],
    symbol_mapping: Dict[str, Symbol],
    config: MistConfig,
    grad_outputs: Union[str, Any] = "auto",
    has_pre_layer: bool = True,
    has_post_layer: bool = True,
) -> List[LayerInfo]:
    """Analyze a list of blocks and return their LayerInfo."""

    # Record the basic information about the layer
    # e.g. inputs and outputs of each node in the computational graph
    recorders = []
    inputs = inputs.values()

    with temporarily_set_sp_eq_ne():
        # * Run the forward pass for the pipe module
        for i, layer in enumerate(blocks):
            recorder = ExecInfoRecorder(layer)
            inputs = recorder.run_forward(*inputs)
            inputs = inputs if isinstance(inputs, tuple) else (inputs,)
            recorders.append(recorder)

        # * Run the backward pass for the pipe module
        if grad_outputs == "auto":
            assert len(inputs) == 1 and isinstance(
                inputs[0], torch.Tensor
            ), f"If grad_outputs is 'auto', inputs must be a tensor, but got {inputs[0]}. Type: {type(inputs[0])}"
            grad_outputs = torch.ones_like(inputs[0])
        for layer, recorder in zip(reversed(blocks), reversed(recorders)):
            grad_outputs = recorder.run_backward(grad_outputs)

    # Analyze the abstract layer
    # We are interested in the last block layer which should be recorders[-2]
    # TODO(zhanda): -2 should be updated
    interesting_recorders = {}
    assert len(list(recorders[-2].gm_nodes_to_recorders.values())) == 1
    interesting_recorders["block_layer"] = next(
        iter(recorders[-2].gm_nodes_to_recorders.values())
    )
    if has_pre_layer:
        interesting_recorders["pre_layer"] = recorders[0]
    if has_post_layer:
        interesting_recorders["post_layer"] = recorders[-1]

    abstract_layer_strategy: LayerStrategy = create_strategy_for_a_layer(
        "abstract_layer", config, mapping=symbol_mapping
    )

    layer_infos: Dict[str, LayerInfo] = {}
    for name, recorder in interesting_recorders.items():
        # Analyze the latency / memory costs for each layer
        layer_analyzer = LayerAnalyzer(recorder, abstract_layer_strategy, config)
        # Run and do the analysis
        layer_info: LayerInfo = layer_analyzer.run()
        layer_info.assert_dataclass_complete()
        layer_infos[name] = layer_info

    return layer_infos
