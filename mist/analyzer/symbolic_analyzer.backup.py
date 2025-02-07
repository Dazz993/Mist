"""Concrete Memory and Performance Analyzer for Hierarchical PyTorch Models"""

from typing import Dict, List, Optional, Tuple, Any, Iterator, Sequence, Union, Callable
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import abstractmethod
from functools import partial

import inspect
import sympy as sp
import torch
from torch import fx
from torch.fx import Interpreter, GraphModule
from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, Node, Target
from torch.utils._pytree import tree_map, tree_flatten

from mist.analyzer.summary import MemorySummarizer, ThroughputSummarizer
from mist.tuning import MistConfig
from mist.logger import get_logger
from mist.memory_pool import (
    MemorySnapshot,
    remove_weights_in_saved_tensors,
    remove_weights_in_saved_tensors_in_snapshot,
    peak_memory_among_different_snapshots,
    compute_memory_for_flattened,
    nbytes,
)
from mist.node_database.database import NodeDB
from mist.node_database.inputs_outputs_spec import (
    map_to_materialized_tensor,
)
from mist.node_database.infer_grad_inputs import infer_grad_inputs_for_symop
from mist.node_database.symbolic_node_spec import SymbolicNodeSpec
from mist.symbols import SymbolManager
from mist.sym_torch.symbolic_tensor import (
    SymbolicTensor,
    SymbolicOpContext,
    SYMOPS,
)
from mist.tuning import MistConfig, MistIntraOpBaseConfig
from mist.tracer.symbolic_tracer import MistSymbolicTracer
from mist.utils.memory import materialize_tensor
from mist.utils.common import map_args_kwargs_to_kwargs, map_args_kwargs_to_args
from mist.utils.sympy import floor_div
from mist.utils.pytree import tree_flatten_like

logger = get_logger()

# db = NodeDB()

# SymbolicShape can be
# 1. str: "b s h"
# 2. Sequence[Union[int, sp.Basic]]: [4, s, h]
SymbolicShape = Sequence[Union[int, sp.Basic, str]]


def map_torch_tensor_to_symbolic_tensor(obj: Union[torch.Tensor, SymbolicTensor, Any]):
    """
    Map a torch tensor to a symbolic tensor
    """
    if isinstance(obj, SymbolicTensor):
        return obj
    elif isinstance(obj, torch.Tensor):
        obj = obj.to("meta")
        return SymbolicTensor(
            obj,
            obj.shape,
        )
    else:
        return obj


class ExecType(Enum):
    """
    Execution Type of the running stage of a node

    1. FWD and CKPT_RECOMPUTE are used in the forward pass while saving saved_tensors.
    2. CKPT_FWD is used in the forward pass while discarding saved_tensors.
    3. BWD is used in the backward pass.
    """

    FWD = auto()
    CKPT = auto()
    BWD = auto()


@dataclass
class ConcreteExecInfo:
    """
    Concrete Execution Information of a node
    """

    node: Node
    exec_type: ExecType
    concrete_args: List[Any] = field(default_factory=list)
    concrete_kwargs: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    context: Optional[SymbolicOpContext] = None
    snapshot_before_release: Optional[MemorySnapshot] = None
    snapshot_after_release: Optional[MemorySnapshot] = None

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


class MistSymbolicAnalyzerBase(Interpreter):
    """
    Mist Symbolic Analyzer used to analyze the base block
    """

    def __init__(
        self,
        module: GraphModule,
        config: MistConfig = None,
        snapshot: MemorySnapshot = None,
    ):
        if not isinstance(module, GraphModule):
            raise TypeError(f"Expected GraphModule, but got {type(module).__name__}")

        # Super init will setup `module`, `submodules`, and `garbage_collect_values`
        super().__init__(module, garbage_collect_values=True)
        self.name = "MistSymbolicAnalyzerBase"

        # Record the execution infomation
        self.fwd_node2info: Dict[Node, ConcreteExecInfo] = OrderedDict()
        self.ckpt_node2info: Dict[Node, ConcreteExecInfo] = OrderedDict()
        self.bwd_node2info: Dict[Node, ConcreteExecInfo] = OrderedDict()
        self.exec_type_to_node_to_info: Dict[ExecType, Dict[Node, ConcreteExecInfo]] = {
            ExecType.FWD: self.fwd_node2info,
            ExecType.CKPT: self.ckpt_node2info,
            ExecType.BWD: self.bwd_node2info,
        }

        # Memory related
        self.init_snapshot = snapshot.copy() if snapshot else MemorySnapshot()

        # Setup the config
        self.setup(config)

    def _process_args_kwargs(self, args, kwargs):
        binded_sig = inspect.signature(self.module.forward).bind(*args, **kwargs)
        args = list(binded_sig.args)
        kwargs = binded_sig.kwargs
        ret = args.copy()
        for i, placeholder in enumerate(self.placeholder_nodes):
            if i < len(args):
                continue
            # Placeholder's name may be changed during the tracing initialization
            # see ``torch/fx/_symbolic_trace.py:Tracer.create_args_for_root``
            # To solve the inconsistency, we try to map the name by guessing.
            if "_" not in placeholder.name:
                name = placeholder.name
            else:
                prefix, _, suffix = placeholder.name.rpartition("_")
                if suffix.isdigit():
                    name = prefix
                else:
                    name = placeholder.name

            if name in kwargs:
                ret.append(kwargs[name])
            else:
                raise RuntimeError(
                    f"Placeholder {placeholder.name} is not provided in the input"
                )
        return ret

    def setup(self, config: MistConfig = None):
        """
        Setup the optimization and execution config. Generate the execution sequence correspondingly.
        """
        self.config = config

        # Collect the input nodes and output nodes
        self.placeholder_nodes = []
        self.output_node = None
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                self.placeholder_nodes.append(node)
            elif node.op == "output":
                assert self.output_node is None, "Multiple output nodes found"
                self.output_node = node

        # Generate the execution sequence
        self.generate_execution_sequence()

    def generate_execution_sequence(self):
        # Generate the execution sequence for forward and backward pass
        self.node_exec_seq_fwd = []
        for node in self.module.graph.nodes:
            self.node_exec_seq_fwd.append(node)

        self.node_exec_seq_bwd = []
        for node in self.generate_backward_sequence():
            self.node_exec_seq_bwd.append(node)

    @abstractmethod
    def run_forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run_backward(self, *args, **kwargs):
        raise NotImplementedError

    def generate_backward_sequence(self):
        """
        Generate the backward node execution sequence based on the forward node execution sequence using topological sort, because the node's backward can only be executed after all its users' backward are executed.
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

    def fetch_args_kwargs_from_node_info(self, n: Node, exec_type: ExecType):
        args = self.map_nodes_to_values(n.args, exec_type)
        assert isinstance(args, tuple), f"args must be a tuple, but got {args}"
        kwargs = self.map_nodes_to_values(n.kwargs, exec_type)
        assert isinstance(kwargs, dict), f"kwargs must be a dict, but got {kwargs}"
        return args, kwargs

    def map_nodes_to_values(self, args: Argument, exec_type: ExecType = ExecType.FWD):
        def load_arg(n_arg: Node) -> Any:
            if n_arg not in self.exec_type_to_node_to_info[exec_type]:
                raise RuntimeError(
                    f"Node {n_arg.format_node()} is not executed in {exec_type}"
                )
            return self.exec_type_to_node_to_info[exec_type][n_arg].output

        return fx.node.map_arg(args, load_arg)

    def log_info(self, concrete_info):
        node = concrete_info.node
        exec_type = concrete_info.exec_type
        snapshot_before_release = concrete_info.snapshot_before_release
        snapshot_after_release = concrete_info.snapshot_after_release

        logger.debug("")
        logger.debug("=" * 80)
        logger.debug(
            f"[{exec_type}] Mod [{self.module.name}] Node [{node.name}]: {node.format_node()}"
        )
        logger.debug("--- snapshot before release:")
        logger.debug(
            f"------ intermediate: {snapshot_before_release.get_category('intermediate')}"
        )
        logger.debug(
            f"------ saved_tensors: {snapshot_before_release.get_category('saved_tensors')}"
        )
        logger.debug("--- snapshot after release:")
        logger.debug(
            f"------ intermediate: {snapshot_after_release.get_category('intermediate')}"
        )
        logger.debug(
            f"------ saved_tensors: {snapshot_after_release.get_category('saved_tensors')}"
        )
        logger.debug("=" * 80)


def _fix_batch_size_dimension(tensor, config):
    if not isinstance(tensor, SymbolicTensor):
        return tensor

    batch_size = config.get("batch_size")
    dp_size = config.get("dp_size")
    shape = list(tensor.shape)
    new_shape = []
    for i, dim in enumerate(shape):
        if isinstance(dim, sp.Basic) and batch_size in dim.free_symbols:
            new_shape.append(floor_div(batch_size, dp_size))
        else:
            new_shape.append(dim)
    new_shape = tuple(new_shape)

    _tensor = tensor.clone()
    _tensor._symbolic_shape = new_shape
    return _tensor


class MistSymbolicAnalyzerForBaseBlock(MistSymbolicAnalyzerBase):
    """
    Mist Symbolic Analyzer used to analyze the base block (i.e. preprocess, block, and postprocess).
    """

    def __init__(
        self,
        module: GraphModule,
        config: MistConfig = None,
        snapshot: MemorySnapshot = None,
    ):
        if not isinstance(config, MistIntraOpBaseConfig):
            raise ValueError(
                f"config must be an instance of MistIntraOpBaseConfig for base block, but got {config}"
            )
        super().__init__(module, config, snapshot)
        self.block_info = None

    def run_forward(self, *args, **kwargs):
        self.cur_snapshot_fwd = self.init_snapshot.copy()
        self.cur_snapshot_ckpt = self.init_snapshot.copy()

        # Processing the input arguments
        # 1. Deal with the batch dimension because different stages may have different micro-batch
        # size due to changed parallelism strategy
        fn_fix_batch_size_dimension = partial(
            _fix_batch_size_dimension, config=self.config
        )
        args = tree_map(fn_fix_batch_size_dimension, args)
        kwargs = tree_map(fn_fix_batch_size_dimension, kwargs)

        # Fix the issue that all tensors are concrete torch tensors
        args = tree_map(map_torch_tensor_to_symbolic_tensor, args)
        kwargs = tree_map(map_torch_tensor_to_symbolic_tensor, kwargs)

        self._args = self._process_args_kwargs(args, kwargs)
        self.args_iter: Iterator[Any] = iter(self._args)

        # For CKPT mode, save the input tensors
        self.cur_snapshot_ckpt.batch_add(
            *tree_flatten(self._args)[0],
            category="saved_tensors",
            comment=f"saved_for_CKPT",
        )

        for node in self.node_exec_seq_fwd:
            # Get the values for each argument of the node
            # It actually doesn't matter whether it is from CKPT or FWD, because the output values are the same
            concrete_args, concrete_kwargs = self.fetch_args_kwargs_from_node_info(
                node, ExecType.FWD
            )

            # Set the default execution info
            self.fwd_node2info[node] = ConcreteExecInfo(
                node=node,
                exec_type=ExecType.FWD,
                concrete_args=concrete_args,
                concrete_kwargs=concrete_kwargs,
            )
            self.ckpt_node2info[node] = ConcreteExecInfo(
                node=node,
                exec_type=ExecType.CKPT,
                concrete_args=concrete_args,
                concrete_kwargs=concrete_kwargs,
            )

            # Execute the node
            output = getattr(self, node.op)(node.target, concrete_args, concrete_kwargs)
            # Used to convert on-the-fly created torch tensors to symbolic tensors
            output = map_torch_tensor_to_symbolic_tensor(output)

            # Save the tensors for the backward pass if needed
            if (
                hasattr(output, "context")
                and output.context is not None
                and not getattr(output.context, "already_saved", False)
            ):
                saved_tensors = output.context.saved_tensors
                if saved_tensors is not None:
                    self.cur_snapshot_fwd.batch_add(
                        *saved_tensors,
                        category="saved_tensors",
                        comment=f"saved_for_{node.name}",
                    )
                output.context.already_saved = True
                output.context.direct_node_parent = node

            # Save the output only in the outer block
            if node.op != "output":
                self.cur_snapshot_fwd.batch_add(
                    *tree_flatten(output)[0], category="intermediate", comment=node.name
                )
                self.cur_snapshot_ckpt.batch_add(
                    *tree_flatten(output)[0], category="intermediate", comment=node.name
                )

            # Release the tensors that are not needed anymore
            snapshot_before_release_fwd = self.cur_snapshot_fwd.copy()
            snapshot_before_release_ckpt = self.cur_snapshot_ckpt.copy()
            for to_delete_node in self.user_to_last_uses.get(node, []):
                to_delete = self.fwd_node2info[to_delete_node].output
                self.cur_snapshot_fwd.batch_remove(
                    *tree_flatten(to_delete)[0], category="intermediate"
                )
                self.cur_snapshot_ckpt.batch_remove(
                    *tree_flatten(to_delete)[0], category="intermediate"
                )

            # Fix the issue that the weights are also added into the saved_tensors
            remove_weights_in_saved_tensors_in_snapshot(
                snapshot_before_release_fwd, self.module
            )
            remove_weights_in_saved_tensors_in_snapshot(
                snapshot_before_release_ckpt, self.module
            )
            remove_weights_in_saved_tensors_in_snapshot(
                self.cur_snapshot_fwd, self.module
            )
            remove_weights_in_saved_tensors_in_snapshot(
                self.cur_snapshot_ckpt, self.module
            )

            # Update the execution info
            self.fwd_node2info[node].update(
                output=output,
                snapshot_before_release=snapshot_before_release_fwd,
                snapshot_after_release=self.cur_snapshot_fwd.copy(),
            )
            self.ckpt_node2info[node].update(
                output=output,
                snapshot_before_release=snapshot_before_release_ckpt,
                snapshot_after_release=self.cur_snapshot_ckpt.copy(),
            )

            self.log_info(self.fwd_node2info[node])
            # self.log_info(self.ckpt_node2info[node])

        output = self.fwd_node2info[self.output_node].output
        return output

    def run_backward(self, grad_output=None):
        if grad_output is None:
            output = self.fwd_node2info[self.output_node].output
            assert (
                isinstance(output, torch.Tensor) and output.numel() == 1
            ), f"grad_output is None, but the output is not a scalar tensor: {output}"
            grad_output = torch.ones_like(output)
        self.bwd_node2info[self.output_node] = ConcreteExecInfo(
            node=self.output_node,
            exec_type=ExecType.BWD,
            output=grad_output,
        )
        self.cur_snapshot_bwd = self.cur_snapshot_fwd.copy()
        self.cur_snapshot_bwd.batch_add(
            grad_output,
            category="intermediate",
            comment=f"grad_output_{self.module.name}",
        )

        for node in self.node_exec_seq_bwd:
            fwd_info = self.fwd_node2info[node]
            bwd_info = self.bwd_node2info.get(node, None)

            # If there is no output for the current node, it is not involved in the
            # backward pass, so we can skip it
            if bwd_info is None or bwd_info.output is None:
                logger.debug("")
                logger.debug("=" * 80)
                logger.debug(
                    'Node [BWD] from ["%s"]: %s has no grad_output',
                    self.module.name,
                    node.format_node(),
                )
                logger.debug("=" * 80)
                continue

            if node.op == "placeholder":
                fulfilled_node_args = []
                fulfilled_grad_args = [bwd_info.output]

            elif node.op == "output":
                fulfilled_node_args = [node.args[0]]
                fulfilled_grad_args = [bwd_info.output]

            elif node.op in {"call_method", "call_module", "call_function"}:
                if node.op == "call_method":
                    target_callable = getattr(
                        fwd_info.concrete_args[0].__class__, node.target
                    )
                    real_target_obj = getattr(torch.Tensor, node.target)
                elif node.op == "call_module":
                    target_callable = self.fetch_attr(node.target)
                    real_target_obj = self.fetch_attr(node.target)
                elif node.op == "call_function":
                    target_callable = node.target
                    real_target_obj = target_callable

                node_spec = SymbolicNodeSpec.from_callable(
                    target_callable, *node.args, **node.kwargs
                )
                signature = node_spec.target_spec.signature

                fulfilled_node_args = map_args_kwargs_to_args(
                    signature,
                    *node.args,
                    **node.kwargs,
                )
                grad_args, grad_kwargs = infer_grad_inputs_for_symop(
                    real_target_obj,
                    bwd_info.output,
                    *fwd_info.concrete_args,
                    **fwd_info.concrete_kwargs,
                )
                fulfilled_grad_args = map_args_kwargs_to_args(
                    signature,
                    *grad_args,
                    **grad_kwargs,
                )

                assert len(fulfilled_grad_args) == len(
                    fulfilled_node_args
                ), f"Lenght mismatch for the grad mapping"

            grad_to_delete = set()
            for arg_node, grad in zip(fulfilled_node_args, fulfilled_grad_args):
                if not isinstance(arg_node, Node):
                    continue
                if not isinstance(grad, torch.Tensor):
                    continue

                # (2) Assign the grad to the corresponding node if not inited
                if arg_node not in self.bwd_node2info:
                    self.bwd_node2info[arg_node] = ConcreteExecInfo(
                        node=arg_node,
                        exec_type=ExecType.BWD,
                        output=grad,
                    )
                else:
                    grad_to_delete.add(arg_node)

                # (3) Add to the memory pool
                self.cur_snapshot_bwd.batch_add(
                    grad,
                    category="intermediate",
                    comment=f"grad_{arg_node.name}",
                )

            # (4) Save the memory snapshot before release
            snapshot_before_release_bwd = self.cur_snapshot_bwd.copy()

            # (5) Release the memory
            # Release the accumulated tmp grads
            for grad in grad_to_delete:
                self.cur_snapshot_bwd.batch_remove(
                    grad,
                    category="intermediate",
                )
            # Release the saved_tensors
            if (
                hasattr(fwd_info.output, "context")
                and fwd_info.output.context is not None
                and getattr(fwd_info.output.context, "direct_node_parent", None) is node
            ):
                saved_tensors = fwd_info.output.context.saved_tensors
                if saved_tensors is not None:
                    saved_tensors = remove_weights_in_saved_tensors(
                        saved_tensors, self.module
                    )
                    self.cur_snapshot_bwd.batch_remove(
                        *saved_tensors,
                        category="saved_tensors",
                    )
            # Release the output grad of the current node
            self.cur_snapshot_bwd.batch_remove(
                bwd_info.output,
                category="intermediate",
            )

            # (6) Update the execution info
            self.bwd_node2info[node].update(
                snapshot_before_release=snapshot_before_release_bwd,
                snapshot_after_release=self.cur_snapshot_bwd.copy(),
            )

            self.log_info(self.bwd_node2info[node])


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


class MistSymbolicAnalyzer(MistSymbolicAnalyzerBase):
    """
    Mist Symbolic Analyzer used to analyze the preprocessing, base block, and the postprocessing
    """

    def __init__(
        self,
        module: GraphModule,
        config: MistConfig = None,
        snapshot: MemorySnapshot = None,
    ):
        super().__init__(module, config, snapshot)
        self.setup_configs()
        self.setup_stages()

    def setup_stages(self):
        self.stages = ["preprocessing", "block", "postprocessing"]
        self.stage_node = {}
        self.stage_module = {}
        self.stage_analyzer = {}
        self.stage_info = {stage: {} for stage in self.stages}

        for node in self.module.graph.nodes:
            op, target, name = node.op, node.target, node.name
            if name in self.stages:
                self.stage_node[name] = node
                self.stage_module[name] = self.fetch_attr(target)
                self.stage_analyzer[name] = MistSymbolicAnalyzerForBaseBlock(
                    self.stage_module[name], self.stage_config[name]
                )

    def setup_configs(self):
        config_for_each_layer = self.config.config_for_each_layer
        self.stage_config = {
            "preprocessing": config_for_each_layer[0],
            "block": config_for_each_layer[1].rename("block"),
            "postprocessing": config_for_each_layer[-1],
        }

    def run(self, *args, **kwargs):
        fwd_output = self.run_forward(*args, **kwargs)
        return fwd_output

    def run_forward(self, *args, **kwargs):
        self._args = self._process_args_kwargs(args, kwargs)
        self.args_iter: Iterator[Any] = iter(self._args)

        self.cur_snapshot_fwd = self.init_snapshot.copy()

        for node in self.node_exec_seq_fwd:
            # Get the values for each argument of the node
            concrete_args, concrete_kwargs = self.fetch_args_kwargs_from_node_info(
                node, ExecType.FWD
            )

            # Set the default execution info
            self.fwd_node2info[node] = ConcreteExecInfo(
                node=node,
                exec_type=ExecType.FWD,
                concrete_args=concrete_args,
                concrete_kwargs=concrete_kwargs,
            )

            # Execute the node
            # It is most likely that the node is a stage node, or it is an auxiliary node
            if node.name in self.stages:
                output = self.stage_analyzer[node.name].run_forward(
                    *concrete_args, **concrete_kwargs
                )
            else:
                output = getattr(self, node.op)(
                    node.target, concrete_args, concrete_kwargs
                )

            # Save the output for the backward pass if needed
            if node.op != "output":
                self.cur_snapshot_fwd.batch_add(
                    *tree_flatten(output)[0], category="intermediate", comment=node.name
                )

            # Release the tensors that are not needed anymore
            snapshot_before_release = self.cur_snapshot_fwd.copy()
            for to_delete_node in self.user_to_last_uses.get(node, []):
                to_delete = self.fwd_node2info[to_delete_node].output
                self.cur_snapshot_fwd.batch_remove(
                    *tree_flatten(to_delete)[0], category="intermediate"
                )

            # Update the execution info
            self.fwd_node2info[node].update(
                output=output,
                snapshot_before_release=snapshot_before_release,
                snapshot_after_release=self.cur_snapshot_fwd.copy(),
            )

            self.log_info(self.fwd_node2info[node])

        return output

    def run_backward(self, grad_output=None):
        if grad_output is None:
            output = self.fwd_node2info[self.output_node].output
            assert (
                isinstance(output, torch.Tensor) and output.numel() == 1
            ), f"grad_output is None, but the output is not a scalar tensor: {output}"
            grad_output = torch.ones_like(output)
        self.bwd_node2info[self.output_node] = ConcreteExecInfo(
            node=self.output_node,
            exec_type=ExecType.BWD,
            output=grad_output,
        )
        self.cur_snapshot_bwd = self.cur_snapshot_fwd.copy()
        self.cur_snapshot_bwd.batch_add(
            grad_output,
            category="intermediate",
            comment=f"grad_output_{self.module.name}",
        )

        for node in self.node_exec_seq_bwd:
            fwd_info = self.fwd_node2info[node]
            bwd_info = self.bwd_node2info.get(node, None)

            # If there is no output for the current node, it is not involved in the
            # backward pass, so we can skip it
            if bwd_info is None or bwd_info.output is None:
                logger.debug("")
                logger.debug("=" * 80)
                logger.debug(
                    'Node [BWD] from ["%s"]: %s has no grad_output',
                    self.module.name,
                    node.format_node(),
                )
                logger.debug("=" * 80)
                continue

            if node.op == "placeholder":
                fulfilled_node_args = []
                fulfilled_grad_args = [bwd_info.output]

            elif node.op == "output":
                fulfilled_node_args = [node.args[0]]
                fulfilled_grad_args = [bwd_info.output]

            elif node.op in {"call_method", "call_module", "call_function"}:
                if node.op == "call_method":
                    target_callable = getattr(
                        fwd_info.concrete_args[0].__class__, node.target
                    )
                    real_target_obj = getattr(torch.Tensor, node.target)
                elif node.op == "call_module":
                    target_callable = self.fetch_attr(node.target).forward
                    real_target_obj = self.fetch_attr(node.target)
                elif node.op == "call_function":
                    target_callable = node.target
                    real_target_obj = target_callable

                fulfilled_node_args = map_args_kwargs_to_args(
                    inspect.signature(target_callable),
                    *node.args,
                    **node.kwargs,
                )
                grad_args, grad_kwargs = infer_grad_inputs_for_symop(
                    real_target_obj,
                    bwd_info.output,
                    *fwd_info.concrete_args,
                    **fwd_info.concrete_kwargs,
                )
                fulfilled_grad_args = map_args_kwargs_to_args(
                    inspect.signature(target_callable),
                    *grad_args,
                    **grad_kwargs,
                )

                assert len(fulfilled_grad_args) == len(
                    fulfilled_node_args
                ), f"Lenght mismatch for the grad mapping"

                if node.op == "call_module":
                    submod = self.fetch_attr(node.target)
                    if isinstance(submod, GraphModule):
                        sub_analyzer = self.stage_analyzer[node.name]
                        sub_analyzer.run_backward(
                            bwd_info.output,
                        )

            grad_to_delete = set()
            for arg_node, grad in zip(fulfilled_node_args, fulfilled_grad_args):
                if not isinstance(arg_node, Node):
                    continue
                if not isinstance(grad, torch.Tensor):
                    continue

                # (2) Assign the grad to the corresponding node if not inited
                if arg_node not in self.bwd_node2info:
                    self.bwd_node2info[arg_node] = ConcreteExecInfo(
                        node=arg_node,
                        exec_type=ExecType.BWD,
                        output=grad,
                    )
                else:
                    grad_to_delete.add(arg_node)

                # (3) Add to the memory pool
                self.cur_snapshot_bwd.batch_add(
                    grad,
                    category="intermediate",
                    comment=f"grad_{arg_node.name}",
                )

            # (4) Save the memory snapshot before release
            snapshot_before_release_bwd = self.cur_snapshot_bwd.copy()

            # (5) Release the memory
            # Release the accumulated tmp grads
            for grad in grad_to_delete:
                self.cur_snapshot_bwd.batch_remove(
                    grad,
                    category="intermediate",
                )

            # Release the output grad of the current node
            self.cur_snapshot_bwd.batch_remove(
                bwd_info.output,
                category="intermediate",
            )

            # (6) Update the execution info
            self.bwd_node2info[node].update(
                snapshot_before_release=snapshot_before_release_bwd,
                snapshot_after_release=self.cur_snapshot_bwd.copy(),
            )

            self.log_info(self.bwd_node2info[node])

    def memory_summary(self):
        self.memory_summarizer = MemorySummarizer(self)
        self.memory_summarizer.forward_summary()
        self.memory_summarizer.backward_summary()

    def throughput_summary(self):
        self.throughput_summarizer = ThroughputSummarizer(self)
        self.throughput_summarizer.summary()
