from typing import Optional, Callable, Union, Tuple, Dict, List, Sequence, Any
import os
import json
import pickle
import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import operator
import sys
import torch
from torch import nn, fx
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from mist.node_database.node_spec import NodeSpec
from mist.node_database.inputs_outputs_spec import (
    InputsSpec,
    TensorSpec,
    OutputSpec,
    UndeterminedOutputSpec,
    EmptyOutputSpec,
)
from mist.node_database.hardware_spec import HardwareSpec, get_hardware_spec
from mist.node_database.saved_tensors_spec import SavedTensorsSpec
from mist.node_database.infer_saved_tensors import infer_saved_tensors
from mist.node_database.infer_grad_inputs import infer_grad_inputs
from mist.node_database.benchmark_computation_latency import benchmark_node
from mist.utils.common import process_benchmarking_results, _format_time
from mist.logger import get_logger

logger = get_logger()

CUR_FILE = Path(__file__).resolve()


@dataclass
class ThroughputResult:
    node_db_table: "NodeDBTable"
    node_spec: NodeSpec
    inputs_spec: InputsSpec
    grad_inputs_spec: Optional[InputsSpec] = None
    output_spec: Optional[OutputSpec] = UndeterminedOutputSpec()
    grad_output_spec: Optional[OutputSpec] = UndeterminedOutputSpec()
    fwd_latency_list: List[float] = field(default_factory=list)
    bwd_latency_list: List[float] = field(default_factory=list)
    fwd_latency: Optional[float] = None
    bwd_latency: Optional[float] = None
    # saved_tensors: SavedTensorsSpec = SavedTensorsSpec()

    @classmethod
    def return_empty(cls):
        return cls(
            node_db_table=None,
            node_spec=None,
            inputs_spec=None,
        )

    @property
    def node_typename(self):
        return self.node_spec.typename if self.node_spec else None

    @property
    def node_instance_spec_str(self):
        return self.node_spec.instance_spec_str if self.node_spec else None

    @property
    def inputs_spec_str(self):
        return str(self.inputs_spec)

    @property
    def existing(self):
        return self.fwd_latency is not None

    def clear(self):
        self.fwd_latency_list = []
        self.bwd_latency_list = []
        self.fwd_latency = None
        self.bwd_latency = None

    def profile(self):
        node_spec = self.node_spec
        inputs_spec = self.inputs_spec
        output_spec = self.output_spec
        grad_output_spec = self.grad_output_spec

        flat_inputs, _flat_spec_inputs = tree_flatten(inputs_spec.inputs)
        flat_grad_output, _flat_spec_grad_output = tree_flatten(grad_output_spec.output)

        # Determine which method to use for profiling
        len_tensor_spec_in_grad_output = len(
            [t for t in flat_grad_output if isinstance(t, TensorSpec)]
        )
        len_requires_grad_in_inputs = len(
            [t for t in flat_inputs if isinstance(t, TensorSpec) and t.requires_grad]
        )
        requires_grad = len_tensor_spec_in_grad_output >= 1 and (
            len_requires_grad_in_inputs >= 1
            or any(r for r in node_spec.param2requires_grad.values())
        )
        can_directly_do_backward = (
            output_spec is not None
            and type(output_spec) != EmptyOutputSpec
            and type(output_spec) != UndeterminedOutputSpec
            and isinstance(output_spec.output, TensorSpec)
        )
        forward_only = not requires_grad or not can_directly_do_backward

        # Latency profiling
        if len_tensor_spec_in_grad_output == 1:
            fwd_latencies, bwd_latencies = benchmark_node(
                self.node_spec, self.inputs_spec, forward_only=forward_only
            )
            (
                fwd_latency_mean,
                fwd_latency_median,
                fwd_latency_std,
            ) = process_benchmarking_results(fwd_latencies)
            (
                bwd_latency_mean,
                bwd_latency_median,
                bwd_latency_std,
            ) = process_benchmarking_results(bwd_latencies)
            self.fwd_latency_list.append(fwd_latencies)
            self.bwd_latency_list.append(bwd_latencies)
            self.fwd_latency = fwd_latency_mean
            self.bwd_latency = bwd_latency_mean

            self.print_readable(print_fn=logger.debug)
            logger.debug(f"--- detailed latency statistics:")
            logger.debug(
                f"------ fwd latency: [MEAN] {_format_time(fwd_latency_mean)}, [MEDIAN] {_format_time(fwd_latency_median)}, [STD] {fwd_latency_std:.8f}"
            )
            logger.debug(
                f"------ bwd latency: [MEAN] {_format_time(bwd_latency_mean)}, [MEDIAN] {_format_time(bwd_latency_median)}, [STD] {bwd_latency_std:.8f}"
            )

        else:
            self.fwd_latency = 0.0
            self.bwd_latency = 0.0

        # Memory profiling
        if not forward_only:
            self.saved_tensors = infer_saved_tensors(self.node_spec, self.inputs_spec)

        if len_tensor_spec_in_grad_output >= 1:
            if len_requires_grad_in_inputs >= 1:
                self.grad_inputs_spec = infer_grad_inputs(
                    self.node_spec, self.inputs_spec, self.grad_output_spec
                )
            else:
                args = tree_map(lambda x: None, inputs_spec.args)
                kwargs = tree_map(lambda x: None, inputs_spec.kwargs)
                self.grad_inputs_spec = InputsSpec(node_spec.signature, *args, **kwargs)

        # Save to database
        self.save()

    def save(self):
        assert self.existing, "Cannot save invalid ExecInfo"
        self.node_db_table.save()

    def __repr__(self) -> str:
        return (
            f"ExecInfo(node_typename={self.node_typename}, "
            f"node_instance_spec_str={self.node_instance_spec_str}, "
            f"inputs_spec_str={self.inputs_spec_str}, "
            f"fwd_latency={self.fwd_latency}, "
            f"bwd_latency={self.bwd_latency})"
        )

    def print_readable(self, print_fn: Callable[[str], None] = print):
        print_fn(f"ExecInfo: '{self.node_typename}'")
        print_fn(f"--- node instance spec: {self.node_instance_spec_str}")
        print_fn(f"--- inputs spec: {self.inputs_spec_str}")
        print_fn(f"--- fwd latency: {_format_time(self.fwd_latency)}")
        print_fn(f"--- bwd latency: {_format_time(self.bwd_latency)}")
        if self.saved_tensors:
            print_fn(f"--- saved tensors:")
            print_fn(
                f"------ saved inputs signature: {self.saved_tensors.saved_inputs_signature}"
            )
            print_fn(f"------ save output: {self.saved_tensors.save_output}")
            print_fn(
                f"------ saved params and buffers signature: {self.saved_tensors.saved_params_and_buffers_signature}"
            )
            print_fn(
                f"------ saved intermediate tensors spec: {self.saved_tensors.saved_intermediate_tensors_spec}"
            )


class NodeDBTable:
    """
    Each Node Database entry contains the information of a node type on a specific hardware.
    """

    def __init__(
        self, typename: str, path: str, data: Dict[str, Dict[str, Any]] = None
    ):
        self.typename = typename
        self.path = path

        # instance_spec_str -> inputs_spec_str -> info
        self.data: Dict[str, Dict[str, Any]] = data or {}

    def get(
        self,
        node_spec: NodeSpec,
        inputs_spec: InputsSpec,
        grad_output_spec: UndeterminedOutputSpec,
    ) -> ThroughputResult:
        node_info = (
            self.data.setdefault(node_spec.instance_spec_str, {})
            .setdefault(
                str(inputs_spec),
                {},
            )
            .setdefault(
                str(grad_output_spec),
                ThroughputResult(
                    self, node_spec, inputs_spec, grad_output_spec=grad_output_spec
                ),
            )
        )
        node_info.node_spec = node_spec
        node_info.inputs_spec = inputs_spec
        node_info.grad_output_spec = grad_output_spec
        return node_info

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)

    @classmethod
    def from_pkl(cls, path: str):
        typename, _, _ = path.rpartition(".")  # "torch.add.pkl" -> "torch.add"
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(typename, path, data)


class NodeDB:
    """
    NodeDB contains all the entries of different NodeSpecs, InputsSpecs, and HardwareSpecs.

    The structure is as follows:
    .                                # Folder: the root of the database ("node_db_data" by default)
    ├── HardwareName/                # Folder: different hardware configurations ("a100_pcie_40gb", "rtx_3090", e.g.)
    │   └── NodeTypeName.pkl         # File: different node types ("torch.add", "torch.nn.modules.linear.Linear", e.g.)

    Inside the json file, it contains the information for all this node type with different (constants, param_requires_grad, inputs_spec).
    """

    SKIP_LIST = {"size", "ones", "zeros", "empty", "full", "randn"}

    def __init__(self, hardware_spec: HardwareSpec = None, folder: str = None) -> None:
        self.hardware_spec = hardware_spec or get_hardware_spec()
        self.folder = folder or os.path.join(
            Path(CUR_FILE).parent.absolute(), "node_db_data", self.hardware_spec.name
        )
        self.initialize()

    def initialize(self):
        """
        Init the database. Create the folder if not exists. Load all the entries from the folder if exists.
        """

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

            # As you can see from the name,
            # node_typename -> node_instance_spec -> inputs_spec -> ExecInfo
            self.node_type_to_table = {}

        else:
            self.node_type_to_table = {}
            for filename in os.listdir(self.folder):
                if filename.endswith(".pkl"):
                    typename = filename[:-4]
                    path = os.path.join(self.folder, filename)
                    self.node_type_to_table[typename] = NodeDBTable.from_pkl(path)

    def get_from_spec(
        self,
        node_spec: NodeSpec,
        inputs_spec: InputsSpec,
        grad_output_spec: OutputSpec = None,
        profile_if_not_existing=True,
    ) -> ThroughputResult:
        """
        Get the node info from the database.
        """

        # Get the node table from the node type name
        node_typename = node_spec.typename
        node_table: NodeDBTable = self.node_type_to_table.setdefault(
            node_typename,
            NodeDBTable(
                node_typename, os.path.join(self.folder, node_typename + ".pkl")
            ),
        )

        # Get the node info from the node table
        node_info = node_table.get(node_spec, inputs_spec, grad_output_spec)

        # If the node info is invalid, profile it
        if profile_if_not_existing and not node_info.existing:
            node_info.profile()

        return node_info

    def get_from_instance(
        self,
        root: Union[nn.Module, Callable],
        *args,
        grad_outputs=None,
        profile_if_not_existing=True,
        **kwargs,
    ):
        """
        A node execution instance is determined by the following:
        1. node_spec: the node type
        2. inputs_spec: the inputs in the forward execution
        3. grad_output_spec: the grad_output in the backward execution
        """

        # Don't profile in several cases
        # (1) the root is in the skip list
        # (2) the output is not a tensor

        if isinstance(root, nn.Module):
            name = root.__class__.__name__
        else:
            name = root.__name__

        if name in self.SKIP_LIST:
            logger.debug(f"Skip {root}")
            return ThroughputResult.return_empty()

        node_spec = NodeSpec.from_callable(root)
        inputs_spec = InputsSpec(node_spec.signature, *args, **kwargs)

        output = root(*args, **kwargs) if output is None else output
        output_spec = OutputSpec(output)

        # During forward execution, we don't know the grad_output_spec exactly.
        # Try to infer it from the output_spec in common cases.
        if grad_output is None:
            num_output_tensors = len(
                [t for t in tree_flatten(output)[0] if isinstance(t, torch.Tensor)]
            )
            if num_output_tensors == 0:
                return ExecInfo.return_empty()
            elif num_output_tensors == 1:
                grad_output_spec = OutputSpec(
                    tree_map(set_to_none_if_not_tensor_spec, output_spec.output)
                )
            else:
                grad_output_spec = UndeterminedOutputSpec()

        else:
            grad_output_spec = OutputSpec(grad_output)

        return self.get_from_spec(
            node_spec,
            inputs_spec,
            output_spec,
            grad_output_spec,
            profile_if_not_existing,
        )


def set_to_none_if_not_tensor_spec(x):
    return None if not isinstance(x, TensorSpec) else x
