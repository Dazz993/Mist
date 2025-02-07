from typing import Callable, Dict, List, Optional, Tuple, Union, Type, Any
import torch
from torch import fx, nn
from torch.utils._pytree import tree_map
import inspect
import operator
from collections import OrderedDict
from dataclasses import dataclass, field

from mist.node_database.node_spec import NodeSpec
from mist.node_database.inputs_outputs_spec import InputsSpec, TensorSpec
from mist.utils.initialization import init_empty_weights
from mist.utils.memory import materialize_module, materialize_tensor
from mist.utils.module import getattr_recursive
from mist.utils.inspect import (
    map_args_kwargs_to_args,
    map_args_kwargs_to_kwargs,
)


def dummy_function(x):
    return x


class SavedTensorsSpec:
    def __init__(
        self,
        node_spec: NodeSpec = None,
        saved_inputs_signature: List[str] = None,
        save_output: bool = False,
        saved_params_and_buffers_signature: List[str] = None,
        saved_intermediate_tensors_spec: List[TensorSpec] = None,
    ):
        # If node_spec is None, create an dummy node_spec which is actually empty.
        if node_spec is None:
            node_spec = NodeSpec("call_function", dummy_function)

        self.node_spec = node_spec
        self.signature = self.node_spec.signature

        self.saved_inputs_signature = saved_inputs_signature or []
        self.save_output = save_output
        self.saved_params_and_buffers_signature = (
            saved_params_and_buffers_signature or []
        )
        self.saved_intermediate_tensors_spec = saved_intermediate_tensors_spec or []

    def get_saved_tensors(
        self,
        instance,
        *args: List[Any],
        output: Any,
        **kwargs: Dict[str, Any],
    ):
        idx_to_all_inputs = {
            idx: arg
            for idx, arg in enumerate(
                map_args_kwargs_to_args(self.signature, *args, **kwargs)
            )
        }
        key_to_all_inputs = map_args_kwargs_to_kwargs(self.signature, *args, **kwargs)

        saved_inputs = []
        for sig in self.saved_inputs_signature:
            if isinstance(sig, int):
                saved_inputs.append(idx_to_all_inputs[sig])
            elif isinstance(sig, str):
                saved_inputs.append(key_to_all_inputs[sig])
            else:
                raise RuntimeError(f"Invalid signature {sig}")

        saved_params_and_buffers = []
        if self.node_spec.op == "call_module":
            for name in self.saved_params_and_buffers_signature:
                saved_params_and_buffers.append(fetch_attr(instance, name))

        saved_intermediate = []
        for spec in self.saved_intermediate_tensors_spec:
            saved_intermediate.append(materialize_tensor(spec, device="meta"))

        saved_tensors = {
            "saved_inputs": saved_inputs,
            "saved_output": [output] if self.save_output else [],
            "saved_params_and_buffers": saved_params_and_buffers,
            "saved_intermediate": saved_intermediate,
        }

        return saved_tensors

    def __repr__(self):
        return f"SavedTensorsSpec({self.node_spec})"
