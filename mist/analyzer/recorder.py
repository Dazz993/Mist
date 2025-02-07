"""Concrete Memory and Performance Analyzer for Hierarchical PyTorch Models"""

from __future__ import annotations
import inspect
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import chain
from typing import Dict, List, Optional, Tuple, Any, Iterator, Sequence, Union, Callable
from copy import deepcopy

import sympy as sp
import torch
from torch import fx
from torch.fx import Interpreter, GraphModule
from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, Node, Target
from torch.utils._pytree import tree_map, tree_flatten
from torch.autograd.graph import saved_tensors_hooks

from mist.logger import get_logger
from mist.node_database.infer_grad_inputs import infer_grad_inputs_for_symop
from mist.utils.pytree import tree_flatten_like, tree_zip_map
from mist.utils.module import getattr_recursive
from mist.utils.tensor_entry import tree_to_entries
from mist.memory_pool import (
    MemoryPool,
    saved_tensors_manager as stm,
)

logger = get_logger()


def get_saved_tensors(node, fwd_info, bwd_info, set_flag_to_context: bool = True):
    """
    We have all the saved tensors rules for symbolic torch native functions.
    But we need to handle the saved tensors for the user-defined autograd Functions.
    """
    output = fwd_info.output

    saved_tensors = None
    if (
        node.op == "call_function"
        and not node.target.__module__.startswith(("torch", "_operator", "builtins"))
        and bwd_info is not None
        and bwd_info.output is not None
    ):
        # Get grad_fn from outputs. We assume outputs is the direct output of an
        # torch.autograd.Function. So all the outputs share the same grad_fn.
        grad_fn = None
        if not isinstance(output, (tuple, list)):
            output = (output,)
        for tensor in output:
            if grad_fn is None and hasattr(tensor, "grad_fn"):
                grad_fn = tensor.grad_fn
            elif grad_fn is not None and hasattr(tensor, "grad_fn"):
                assert (
                    grad_fn is tensor.grad_fn
                ), f"Multiple grad_fn found in the outputs"
        

        # Get the saved_tensors from grad_fn
        # When calling ``grad_fn.saved_tensors``, stm will record the saved_tensors
        stm.start_recording(clear=True)

        # Get the saved_tensors from grad_fn
        # ``dir`` is used because ``hasattr`` will call saved_tensors and cause
        # stm to grow.
        # Try to find the correct grad_fn if it's an alias backward grad fn
        assert grad_fn is not None
        if "saved_tensors" in dir(grad_fn):
            raw_saved_tensors = list(
                s for s in grad_fn.saved_tensors if isinstance(s, torch.Tensor)
            )
        elif "alias" in repr(grad_fn).lower() or "view" in repr(grad_fn).lower():
            # Find the correct grad_fn without alias
            while "alias" in repr(grad_fn).lower() or "view" in repr(grad_fn).lower():
                grad_fn = grad_fn.next_functions[0][0]
            assert "saved_tensors" in dir(grad_fn), f"{grad_fn=}, {grad_fn.next_functions=}, {dir(grad_fn)=}"
            raw_saved_tensors = list(
                s for s in grad_fn.saved_tensors if isinstance(s, torch.Tensor)
            )
        elif "_saved_self" in dir(grad_fn) or "_saved_other" in dir(grad_fn):
            raw_saved_tensors = []
            if "_saved_self" in dir(grad_fn) and isinstance(
                grad_fn._saved_self, torch.Tensor
            ):
                raw_saved_tensors.append(grad_fn._saved_self)
            if "_saved_other" in dir(grad_fn) and isinstance(
                grad_fn._saved_other, torch.Tensor
            ):
                raw_saved_tensors.append(grad_fn._saved_other)
        else:
            raise RuntimeError(f"Unknown grad_fn: {grad_fn}, Node: {node}, {dir(grad_fn)=}")

        # Get the saved_tensors from the stm
        saved_tensors = stm.saved_tensors

        # Check the consistency
        assert len(raw_saved_tensors) == len(saved_tensors), f"{len(raw_saved_tensors)} != {len(saved_tensors)}. Grad_fn: {grad_fn}. {raw_saved_tensors=}, {saved_tensors=}"
        stm.stop_recording()

    elif (
        hasattr(output, "context")
        and (context := output.context) is not None
        and not getattr(context, "already_saved", False)
    ):
        saved_tensors = context.saved_tensors
        if set_flag_to_context:
            output.context.already_saved = True

    return saved_tensors


def get_extra_inner_tensors(node, fwd_info, bwd_info):
    output = fwd_info.output
    extra_inner_for_fwd = None
    extra_inner_for_bwd = None
    if (
        hasattr(output, "context")
        and (context := output.context) is not None
        and (context.direct_producer_node == node)
    ):
        extra_inner_for_fwd = context.extra_inner_for_fwd
        extra_inner_for_bwd = context.extra_inner_for_bwd
    return extra_inner_for_fwd, extra_inner_for_bwd


def has_torch_tensor(obj):
    return any(isinstance(item, torch.Tensor) for item in tree_flatten(obj)[0])


def _update_grad_fn(ori_grad, new_grad):
    if ori_grad is None:
        return new_grad
    elif new_grad is None:
        return ori_grad
    else:
        assert isinstance(ori_grad, torch.Tensor)
        assert isinstance(new_grad, torch.Tensor)
        ori_grad += new_grad
        return ori_grad


def _check_grad_fn(input, grad):
    if isinstance(grad, torch.Tensor):
        assert isinstance(input, torch.Tensor)
        assert (
            grad.shape == input.shape
        ), f"input.shape={input.shape} != grad.shape={grad.shape}"
        assert (
            grad.dtype == input.dtype
        ), f"input.dtype={input.dtype} != grad.dtype={grad.dtype}"
        assert (
            grad.device == input.device
        ), f"input.device={input.device} != grad.device={grad.device}"


class ExecType(Enum):
    """
    Execution Type of the running stage of a node
    """

    FWD = auto()
    BWD = auto()


@dataclass
class ExecInfo:
    """
    Execution Information of a node
    """

    # Common
    node: Node
    exec_type: ExecType

    # For recorder
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    saved_tensors = None
    extra_inner_for_fwd: List[Any] = None
    extra_inner_for_bwd: List[Any] = None

    # For analyzer
    memory_pool_before_release: Optional[MemoryPool] = None
    memory_pool_after_release: Optional[MemoryPool] = None

    @property
    def op(self):
        """
        Get the operation of the node
        """
        return self.node.op

    @property
    def target(self):
        return self.node.target

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ExecInfoRecorder(Interpreter):
    """
    A Recorder to record the execution information of a symbolic model
    """

    def __init__(
        self,
        module: GraphModule,
    ):
        if not isinstance(module, GraphModule):
            raise TypeError(f"Expected GraphModule, but got {type(module).__name__}")

        # Super init will setup `module`, `submodules`, and `garbage_collect_values`
        super().__init__(module, garbage_collect_values=True)

        # TODO(zhanda): disable printings
        logger.info(f"Print the graph of module {module.name}")
        module.graph.print_tabular()
        print("")

        # Record the execution infomation
        self.fwd_node2info: Dict[Node, ExecInfo] = OrderedDict()
        self.bwd_node2info: Dict[Node, ExecInfo] = OrderedDict()
        self.exec_type_to_node_to_info: Dict[ExecType, Dict[Node, ExecInfo]] = {
            ExecType.FWD: self.fwd_node2info,
            ExecType.BWD: self.bwd_node2info,
        }

        self.gm_nodes_to_recorders: Dict[Node, ExecInfoRecorder] = {}

        # Setup
        self.setup()

    def setup(self):
        """
        Setup the optimization and execution config. Generate the execution sequence correspondingly.
        """
        # Collect the input nodes and output nodes
        self.name2node = OrderedDict()
        self.placeholder_nodes = []
        self.output_node = None
        for node in self.module.graph.nodes:
            self.name2node[node.name] = node
            if node.op == "placeholder":
                self.placeholder_nodes.append(node)
            elif node.op == "output":
                assert self.output_node is None, "Multiple output nodes found"
                self.output_node = node

        # Generate the execution sequence
        self.generate_execution_sequence()

    def call_function(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        with saved_tensors_hooks(pack_hook=stm.pack_hook, unpack_hook=stm.unpack_hook):
            ret = super().call_function(target, args, kwargs)
        return ret

    def _process_args_kwargs(self, args, kwargs):
        bound_sig = inspect.signature(self.module.forward).bind(*args, **kwargs)
        args = bound_sig.args
        kwargs = bound_sig.kwargs

        # Placeholder's name may be changed during the tracing initialization
        # see ``torch/fx/_symbolic_trace.py:Tracer.create_args_for_root``
        # To solve the inconsistency, we try to map the name by guessing.
        collected_args = list(args)
        for i, placeholder in enumerate(self.placeholder_nodes):
            if i < len(args):
                continue
            if "_" not in placeholder.name:
                name = placeholder.name
            else:
                prefix, _, suffix = placeholder.name.rpartition("_")
                if suffix.isdigit():
                    name = prefix
                else:
                    name = placeholder.name

            if name in kwargs:
                collected_args.append(kwargs[name])
            else:
                raise RuntimeError(
                    f"Placeholder {placeholder.name} is not provided in the input"
                )

        return collected_args

    def run_forward(self, *args, **kwargs):
        self._args = self._process_args_kwargs(args, kwargs)
        self.args_iter: Iterator[Any] = iter(self._args)

        for node in self.node_exec_seq_fwd:
            # Map nodes in node.args and node.kwargs to concrete values
            args, kwargs = self.fetch_args_kwargs_from_node_info(node, ExecType.FWD)
            # Execute the node
            if node.op == "call_module" and isinstance(
                sub_module := getattr_recursive(self.module, node.target), GraphModule
            ):
                sub_module_recorder = ExecInfoRecorder(sub_module)
                output = sub_module_recorder.run_forward(*args, **kwargs)
                self.gm_nodes_to_recorders[node] = sub_module_recorder
            else:
                output = getattr(self, node.op)(node.target, args, kwargs)
            # Record the info
            self.fwd_node2info[node] = ExecInfo(
                node=node,
                exec_type=ExecType.FWD,
                args=args,
                kwargs=kwargs,
                output=output,
            )
            # Update the OpContext
            if getattr(output, "context", None) is not None:
                context = output.context
                if context.direct_producer_node is None:
                    context.direct_producer_node = node

            self.log_info(self.fwd_node2info[node])

        output = self.fwd_node2info[self.output_node].output
        return output

    def run_backward(self, grad_output=None):
        if grad_output is None:
            output = self.fwd_node2info[self.output_node].output
            assert (
                isinstance(output, torch.Tensor) and output.numel() == 1
            ), f"grad_output is None, but the output is not a scalar tensor: {output}"
            grad_output = torch.ones_like(output)

        self.bwd_node2info[self.output_node] = ExecInfo(
            node=self.output_node,
            exec_type=ExecType.BWD,
            output=grad_output,
        )

        for node in self.node_exec_seq_bwd:
            fwd_info = self.fwd_node2info[node]
            bwd_info = self.bwd_node2info.get(node, None)

            if bwd_info is None or bwd_info.output is None:
                logger.debug(
                    f"[ExecType.BWD] Mod ['{self.module.name}'] Node [{node.name}]: {node.format_node()}"
                )
                logger.debug("--- [SKIP BACKWARD]")
                continue

            # Get grads of args and kwargs according to the node's op
            grads_of_args = ()
            grads_of_kwargs = {}

            if node.op == "placeholder":
                pass
            elif node.op == "output":
                grads_of_args = (bwd_info.output,)
                # grads_of_args = (
                #     bwd_info.output
                #     if isinstance(bwd_info.output, tuple)
                #     else (bwd_info.output,)
                # )
            elif node in self.gm_nodes_to_recorders:
                sub_module_recorder = self.gm_nodes_to_recorders[node]
                grads_of_inputs = sub_module_recorder.run_backward(bwd_info.output)
                bound_sig = inspect.signature(sub_module_recorder.module.forward).bind(
                    *grads_of_inputs
                )
                grads_of_args = grads_of_inputs[: len(node.args)]
                grads_of_kwargs = {k: bound_sig.arguments[k] for k in node.kwargs}
            elif node.op in {"call_method", "call_module", "call_function"}:
                grads_of_args, grads_of_kwargs = infer_grad_inputs_for_symop(
                    in_args=fwd_info.args,
                    in_kwargs=fwd_info.kwargs,
                    outputs=fwd_info.output,
                    grad_outputs=bwd_info.output,
                )
                assert len(grads_of_args) == len(node.args)
                assert len(grads_of_kwargs) == len(node.kwargs)
            else:
                raise RuntimeError(f"Unknown op {node.op}")

            self.bwd_node2info[node].update(
                args=grads_of_args,
                kwargs=grads_of_kwargs,
            )

            # Map grad tensors to the corresponding input nodes
            flat_arg_nodes, flat_arg_nodes_spec = tree_flatten(node.args)
            flat_kwarg_nodes, flat_kwarg_nodes_spec = tree_flatten(node.kwargs)
            flat_arg_grads, _ = tree_flatten_like(grads_of_args, flat_arg_nodes_spec)
            flat_kwarg_grads, _ = tree_flatten_like(
                grads_of_kwargs, flat_kwarg_nodes_spec
            )

            flat_input_nodes = flat_arg_nodes + flat_kwarg_nodes
            flat_input_grads = flat_arg_grads + flat_kwarg_grads
            if len(flat_input_nodes) < len(flat_input_grads):
                flat_input_nodes.extend(
                    [None] * (len(flat_input_grads) - len(flat_input_nodes))
                )
            assert len(flat_input_nodes) == len(flat_input_grads)

            for input_node, input_grad in zip(flat_input_nodes, flat_input_grads):
                if not isinstance(input_node, Node):
                    continue
                if not has_torch_tensor(input_grad):
                    continue

                # Check the shape, dtype, and device of the input and the grad
                input = self.fwd_node2info[input_node].output
                tree_zip_map(_check_grad_fn, input, input_grad)

                if input_node not in self.bwd_node2info:
                    self.bwd_node2info[input_node] = ExecInfo(
                        node=input_node,
                        exec_type=ExecType.BWD,
                        output=input_grad,
                    )
                else:
                    self.bwd_node2info[input_node].output = tree_zip_map(
                        fn=_update_grad_fn,
                        pytree_1=self.bwd_node2info[input_node].output,
                        pytree_2=input_grad,
                    )

            self.log_info(self.bwd_node2info[node])

        grad_inputs = tuple(
            self.bwd_node2info[node].output if node in self.bwd_node2info else None
            for node in self.placeholder_nodes
        )
        if len(grad_inputs) == 1 and not isinstance(grad_inputs[0], tuple):
            grad_inputs = grad_inputs[0]

        # Run rerun_for_saved_tensors to get the saved_tensors
        self.rerun_for_saved_tensors()

        return grad_inputs

    def rerun_for_saved_tensors(self):
        for recorder in self.gm_nodes_to_recorders.values():
            recorder.rerun_for_saved_tensors()

        for node in self.node_exec_seq_fwd:
            fwd_info = self.fwd_node2info[node]
            bwd_info = self.bwd_node2info.get(node, None)

            saved_tensors = get_saved_tensors(
                node, fwd_info, bwd_info, set_flag_to_context=False
            )
            extra_inner_for_fwd, extra_inner_for_bwd = get_extra_inner_tensors(
                node, fwd_info, bwd_info
            )
            fwd_info.saved_tensors = saved_tensors
            fwd_info.extra_inner_for_fwd = extra_inner_for_fwd
            fwd_info.extra_inner_for_bwd = extra_inner_for_bwd
            if bwd_info is not None:
                bwd_info.saved_tensors = saved_tensors
                bwd_info.extra_inner_for_fwd = extra_inner_for_fwd
                bwd_info.extra_inner_for_bwd = extra_inner_for_bwd
            self.log_info(fwd_info, "RERUN_FOR_SAVED_TENSORS")

    def generate_execution_sequence(self):
        # Generate the execution sequence for forward and backward pass
        self.node_exec_seq_fwd = []
        for node in self.module.graph.nodes:
            self.node_exec_seq_fwd.append(node)

        self.node_exec_seq_bwd = []
        for node in self.generate_backward_sequence():
            self.node_exec_seq_bwd.append(node)

    def generate_backward_sequence(self):
        """
        Generate the backward node execution sequence based on the forward node
        execution sequence using topological sort, because the node's backward
        can only be executed after all its users' backward are executed.
        """

        node2num_users = {node: len(node.users) for node in self.module.graph.nodes}
        to_pop = []
        for node, num_user in node2num_users.items():
            if num_user == 0:
                if node.op == "output":
                    continue
                elif node.op == "placeholder":
                    to_pop.append(node)
                    continue
                else:
                    raise RuntimeError(
                        f"Node {node} has no users but is not an output node or a placeholder node, should be eliminated by DCE."
                    )
        for node in to_pop:
            node2num_users.pop(node)

        # Topological sort
        # Add the output node to the stack
        stack: List[Node] = [self.output_node]
        bwd_order: List[Node] = []
        while stack:
            node = stack.pop()
            bwd_order.append(node)

            for input_node in node._input_nodes:
                node2num_users[input_node] -= 1
                if node2num_users[input_node] == 0:
                    stack.append(input_node)

        assert len(bwd_order) == len(
            node2num_users
        ), f"len(bwd_order) != len(self.module.graph.nodes), {len(bwd_order)} != {len(self.module.graph.nodes)}"

        return bwd_order

    # TODO(zhanda): can be simplified
    def fetch_args_kwargs_from_node_info(self, n: Node, exec_type: ExecType):
        args = self.map_nodes_to_values(n.args, exec_type)
        assert isinstance(args, tuple), f"args must be a tuple, but got {args}"
        kwargs = self.map_nodes_to_values(n.kwargs, exec_type)
        assert isinstance(kwargs, dict), f"kwargs must be a dict, but got {kwargs}"
        return args, kwargs

    # TODO(zhanda): can be simplified
    def map_nodes_to_values(self, args: Argument, exec_type: ExecType = ExecType.FWD):
        def load_arg(n_arg: Node) -> Any:
            if n_arg not in self.exec_type_to_node_to_info[exec_type]:
                raise RuntimeError(
                    f"Node {n_arg.format_node()} is not executed in {exec_type}"
                )
            return self.exec_type_to_node_to_info[exec_type][n_arg].output

        return fx.node.map_arg(args, load_arg)

    def log_info(self, info, exec_type: ExecType = None):
        node = info.node
        exec_type = exec_type or info.exec_type
        prefix = "grad_" if exec_type == ExecType.BWD else ""

        logger.debug(
            f"[{exec_type}] Mod ['{self.module.name}'] Node [{node.name}]: {node.format_node()}"
        )
        logger.debug(f"--- {prefix}args: {tree_to_entries(info.args)}")
        logger.debug(f"--- {prefix}kwargs: {tree_to_entries(info.kwargs)}")
        logger.debug(f"--- {prefix}output: {tree_to_entries(info.output)}")
        logger.debug(f"--- saved_tensors: {tree_to_entries(info.saved_tensors)}")
        if info.extra_inner_for_fwd is not None:
            logger.debug(
                f"--- extra_inner_for_fwd: {tree_to_entries(info.extra_inner_for_fwd)}"
            )
        if info.extra_inner_for_bwd is not None:
            logger.debug(
                f"--- extra_inner_for_bwd: {tree_to_entries(info.extra_inner_for_bwd)}"
            )
