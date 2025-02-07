from typing import Dict, OrderedDict, List, Set, Tuple
from functools import partial
from pprint import pprint, pformat

import torch
import torch.fx
from torch.fx import Node
from torch.utils._pytree import tree_flatten

from mist.analyzer.recorder import ExecType, ExecInfo, ExecInfoRecorder
from mist.analyzer.strategy import LayerStrategy
from mist.memory_pool import MemoryPool, saved_tensors_manager as stm
from mist.logger import get_logger
from mist.utils.tensor_entry import tree_to_entries, get_tensor_base_id
from mist.node_database.infer_grad_inputs import infer_grad_inputs_for_symop
from mist.utils.pytree import tree_flatten_like, tree_zip_map

logger = get_logger(__name__)


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


class ThroughputAnalyzer:
    def __init__(self, recorder: ExecInfoRecorder, layer_strategy: LayerStrategy):
        self.recorder: ExecInfoRecorder = recorder
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

        self.layer_strategy: LayerStrategy = layer_strategy

        self.memory_pool: MemoryPool = MemoryPool()

        # Auxiliary attributes
        self.params_requires_grad: List[torch.Tensor] = [
            p for p in self.module.parameters() if p.requires_grad
        ]
        self.params_no_requires_grad: List[torch.Tensor] = [
            p for p in self.module.parameters() if not p.requires_grad
        ]
        self.buffers: List[torch.Tensor] = list(self.module.buffers())
        self.exclude_module_attr = partial(
            exclude_tensors_using_base_id,
            exclude_tensors=self.params_requires_grad
            + self.params_no_requires_grad
            + self.buffers,
        )

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

            # Update the execution info
            self.fwd_node2info[node].update(
                memory_pool_before_release=memory_pool_before_release,
                memory_pool_after_release=memory_pool.copy(),
            )

            self.log_info(fwd_info, "MemAnalyzer.FWD")

    def run_backward(self):
        memory_pool = self.memory_pool
        visited_node_grads: Set[Tuple[Node, int]] = set()

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

    def log_info(self, info, exec_type=None):
        node = info.node
        exec_type = exec_type or info.exec_type
        prefix = "grad_" if exec_type == ExecType.BWD else ""
        pool_before_release = info.memory_pool_before_release
        pool_after_release = info.memory_pool_after_release

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
