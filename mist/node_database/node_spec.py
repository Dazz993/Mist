from __future__ import annotations
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Union, Type, Any
import types
import inspect
import operator
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial

# import json
from pathlib import Path
import numpy as np

import sympy as sp
import torch
from torch import fx, nn
from torch.utils._pytree import tree_map
import torch.nn as nn
from torch.utils._pytree import tree_flatten

import mist
from mist.node_database.tensor_spec import TensorSpec
from mist.utils.module import getattr_recursive
from mist.sym_torch.symbolic_tensor import SymbolicTensor
from mist.utils.inspect import inspect_torch_function_signature
from mist.node_database.benchmark_computation_latency import benchmark_node
from mist.utils.common import (
    process_benchmarking_results,
    _format_time,
    save_json,
    save_json_with_lock,
    load_json_with_lock,
)
from mist.logger import get_logger
from mist.memory_pool import nbytes
from mist.tools.optimize import predict_gpu_gpu_comm_latency

logger = get_logger()
CUR_FOLDER = Path(__file__).resolve().parent
MATERIALIZE_RAND_TENSORS = True

GB = 1 << 30

MAX_VALUE = 1e5

def is_module_symbolic(module: nn.Module):
    for constant_name in getattr(module, "__constants__", []):
        constant = getattr(module, constant_name)
        if isinstance(constant, sp.Basic):
            return True
    for param in module.parameters():
        if isinstance(param, SymbolicTensor):
            return True
    for buffer in module.buffers():
        if isinstance(buffer, SymbolicTensor):
            return True
    for item in module.__dict__.values():
        if isinstance(item, SymbolicTensor):
            return True
        if isinstance(item, nn.Module) and is_module_symbolic(item):
            return True
    return False


def map_to_tensor_spec(x):
    if isinstance(x, torch.Tensor):
        return TensorSpec.from_tensor(x)
    return x


def map_to_materialized_tensor(x, device="meta", rand=True):
    if isinstance(x, TensorSpec):
        return x.materialize(device, rand=rand)
    return x


def is_function(fn):
    """
    Judge whether a function is a function or not.

    1. A normal function is a function.
    2. A torch function is a function (e.g. torch.add, torch.nn.functional.linear).
        -> (Built-in, fn.__self__ is None, fn.__module__ starts with "torch")
    3. A built-in function is a function (e.g. operator.add, operator.mul).
        -> (Built-in, fn.__module__ is "_operator")
    """
    return inspect.isfunction(fn) or (
        inspect.isbuiltin(fn)
        and (fn.__self__ is None or isinstance(fn.__self__, types.ModuleType))
        and fn.__module__.startswith(("torch", "_operator"))
    )


def is_method(fn):
    """
    Judge whether a function is a method or not.

    1. A method is a method.
    2. A torch method is a method (e.g. tensor.add).
    """
    return inspect.ismethod(fn) or (
        inspect.isbuiltin(fn) and isinstance(fn.__self__, torch.Tensor)
    )


def is_methoddescriptor(fn):
    return inspect.ismethoddescriptor(fn)


# def predict_comm_latency(op_name, tensor_bytes, gpu_gpu_bandwidth):
#     """bandwidth in GB/s"""
#     gpu_gpu_bandwidth = gpu_gpu_bandwidth * GB
#     if op_name == "reduce_scatter" or op_name == "all_gather":
#         return tensor_bytes / gpu_gpu_bandwidth
#     elif op_name == "all_reduce":
#         return tensor_bytes / gpu_gpu_bandwidth * 2
#     elif op_name == "none":
#         return 0
#     else:
#         raise NotImplementedError(
#             f"Cannot predict the communication latency for {op_name}"
#         )


# def predict_comm_latency(
#     op_name: str,
#     full_tensor_size: Union[np.ndarray, int],
#     inter_size: Union[np.ndarray, int],
#     intra_size: Union[np.ndarray, int],
#     inter_node_bandwidth: float,
#     inter_node_bias: float,
#     inter_node_device_bias: float,
#     intra_node_bandwidth: float,
#     intra_node_bias: float,
#     intra_node_device_bias: float,
# ):
#     """
#     tensor size is in B
#     bandwidth is in GB/s
#     """
#     inter_node_bw = inter_node_bandwidth * GB
#     intra_node_bw = intra_node_bandwidth * GB

#     # Mesh shape is (outer_group_size, inner_group_size)
#     # if inner_group_size != 1, then we need to do comm in intra-node
#     # if outer_group_size != 1, then we need to do comm in inter-node
#     empirical_mode = True
#     if not empirical_mode:
#         inter_comm = (
#             full_tensor_size / inter_node_bandwidth * (inter_size - 1) / inter_size
#         )
#         intra_comm = (
#             full_tensor_size / intra_node_bandwidth * (intra_size - 1) / intra_size
#         )
#     else:
#         inter_comm = (
#             full_tensor_size
#             / inter_node_bandwidth
#             * (inter_size - 1 + inter_node_device_bias)
#             / (inter_size + inter_node_device_bias)
#         ) + inter_node_bias
#         inter_comm = np.where(inter_size == 1, 0, inter_comm)
#         intra_comm = (
#             full_tensor_size
#             / intra_node_bandwidth
#             * (intra_size - 1 + intra_node_device_bias)
#             / (intra_size + intra_node_device_bias)
#         ) + intra_node_bias
#         intra_comm = np.where(intra_size == 1, 0, intra_comm)
#     comm = inter_comm + intra_comm

#     if op_name == "all_reduce":
#         factor = 2
#     elif op_name == "all_gather":
#         factor = 1
#     elif op_name == "reduce_scatter":
#         factor = 1
#     elif op_name == "none":
#         factor = 0
#     else:
#         raise NotImplementedError

#     return comm * factor


def get_counterpart_op_name(op_name):
    if op_name == "reduce_scatter":
        return "all_gather"
    elif op_name == "all_gather":
        return "reduce_scatter"
    elif op_name == "all_reduce":
        return "none"
    elif op_name == "none":
        return "all_reduce"
    else:
        raise NotImplementedError(
            f"Cannot predict the communication latency for {op_name}"
        )


class NodeSpec:
    def __init__(self, target_spec, *args, **kwargs):
        self.target_spec = target_spec

        args_spec = tree_map(map_to_tensor_spec, args)
        kwargs_spec = tree_map(map_to_tensor_spec, kwargs)
        bounded_signature = target_spec.signature.bind(*args_spec, **kwargs_spec)
        bounded_signature.apply_defaults()
        # Bounded signature is used for identifying the node.
        self.bounded_signature = bounded_signature
        # Args and kwargs are used for concretizing the node.
        self.args = args_spec
        self.kwargs = kwargs_spec

    def materialize(self, device="cuda"):
        fn = self.materialize_target(device)
        args, kwargs = self.materialize_inputs(device)
        return fn, args, kwargs

    def materialize_target(self, device="cuda"):
        fn = self.target_spec.materialize(device)
        return fn

    def materialize_inputs(self, device="cuda"):
        _map_to_materialized_tensor = partial(
            map_to_materialized_tensor, device=device, rand=MATERIALIZE_RAND_TENSORS
        )
        args = tree_map(_map_to_materialized_tensor, self.args)
        kwargs = tree_map(_map_to_materialized_tensor, self.kwargs)
        return args, kwargs

    def profile(self, device="cuda", **kwargs):
        device_name = torch.cuda.get_device_name(device).replace(" ", "_")
        cached_folder_path = os.path.join(CUR_FOLDER, f"cache_{device_name}")
        if not os.path.exists(cached_folder_path):
            os.makedirs(cached_folder_path)

        target_spec_identifier = self.target_spec.identifier.split(" ")[0]
        self_repr = self.__repr__()
        cached_file_path = os.path.join(
            cached_folder_path, f"{target_spec_identifier}.json"
        )

        zeros = (0, 0, 0)
        directly_return = False
        extra_fwd_latency, extra_bwd_latency = 0, 0
        if "MistProcessGroup" in repr(self):
            # Find the mist process group
            process_group = []
            process_group.extend(
                [arg for arg in self.args if "MistProcessGroup" in repr(arg)]
            )
            process_group.extend(
                [arg for arg in self.kwargs.values() if "MistProcessGroup" in repr(arg)]
            )
            assert len(process_group) == 1, f"Cannot find the mist process group"
            process_group = process_group[0]

            if self.target_spec.identifier.startswith("mist.distributed.op"):
                directly_return = True
                fwd_op_name = self.target_spec.identifier.split(".")[-1]
                bwd_op_name = get_counterpart_op_name(fwd_op_name)
                tensor_gbytes = (
                    nbytes(list(self.bounded_signature.arguments.values())[0]) / GB
                )
            elif (
                self.target_spec.identifier
                == "mist.modules.fused_dense.fused_dense_func"
            ):
                directly_return = False
                fwd_op_name = (
                    "all_gather" if self.kwargs["sequence_parallel"] else "none"
                )
                bwd_op_name = "all_reduce"
                tensor_gbytes = (
                    nbytes(list(self.bounded_signature.arguments.values())[0]) / GB
                )
            elif (
                self.target_spec.identifier
                == "mist.modules.losses._vocab_parallel_cross_entropy"
            ):
                fwd_op_name = "all_reduce"
                bwd_op_name = "none"
                logits_tensor = list(self.bounded_signature.arguments.values())[0]
                assert len(logits_tensor.shape) == 2
                logits_samples = logits_tensor.shape[0]
                tensor_gbytes = (
                    3
                    * logits_samples
                    * torch.empty([], dtype=logits_tensor.dtype).element_size()
                ) / GB
            else:
                raise NotImplementedError(f"Please register the comm info for {self}")

            if process_group.size() != 1:
                inter_size = kwargs.get("inter_size", None)
                intra_size = kwargs.get("intra_size", None)
                gpu_gpu_comm_params = kwargs.get("gpu_gpu_comm_params", [])
                assert process_group.size() == inter_size * intra_size

                extra_fwd_latency = predict_gpu_gpu_comm_latency(
                    op_name=fwd_op_name,
                    gbytes=tensor_gbytes,
                    inter_size=inter_size,
                    intra_size=intra_size,
                    gpu_gpu_comm_params=gpu_gpu_comm_params,
                )
                extra_bwd_latency = predict_gpu_gpu_comm_latency(
                    op_name=bwd_op_name,
                    gbytes=tensor_gbytes,
                    inter_size=inter_size,
                    intra_size=intra_size,
                    gpu_gpu_comm_params=gpu_gpu_comm_params,
                )

            # To remove the non-existing kernel launch latency
            if directly_return:
                return zeros, zeros, extra_fwd_latency, extra_bwd_latency

        cached = False
        if os.path.exists(cached_file_path):
            results = load_json_with_lock(cached_file_path, logging_level=logging.ERROR)
            if self_repr in results:
                # logger.debug(f"Loaded cached results from {cached_file_path}")
                cached = True
                fwd_latency_mean, fwd_latency_median, fwd_latency_std = results[
                    self_repr
                ]["fwd"]
                bwd_latency_mean, bwd_latency_median, bwd_latency_std = results[
                    self_repr
                ]["bwd"]

        if not cached:
            (fwd_latency_mean, fwd_latency_median, fwd_latency_std), (
                bwd_latency_mean,
                bwd_latency_median,
                bwd_latency_std,
            ) = self._profile(device)

            def update_fn(data):
                data[self_repr] = {
                    "fwd": (
                        fwd_latency_mean,
                        fwd_latency_median,
                        fwd_latency_std,
                    ),
                    "bwd": (
                        bwd_latency_mean,
                        bwd_latency_median,
                        bwd_latency_std,
                    ),
                }
                data = {k: data[k] for k in sorted(data.keys())}
                return data

            save_json_with_lock(cached_file_path, update_fn)

        return (
            (
                fwd_latency_mean,
                fwd_latency_median,
                fwd_latency_std,
            ),
            (
                bwd_latency_mean,
                bwd_latency_median,
                bwd_latency_std,
            ),
            extra_fwd_latency,
            extra_bwd_latency,
        )

    def _profile(self, device="cuda"):
        # Determine whether to benchmarking both forward and backward.
        flat_inputs = tree_flatten(self.bounded_signature.arguments)[0]
        fn, args, kwargs = self.materialize(device)
        try:
            outputs = fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to run {self} on {device}")
            logger.error(f"Function: {fn}")
            # logger.error(f"Args: {args}")
            # logger.error(f"Kwargs: {kwargs}")
            # raise e
            return (
                MAX_VALUE,
                MAX_VALUE,
                0,
            ), (
                MAX_VALUE,
                MAX_VALUE,
                0,
            )   
        flat_outputs = tree_flatten(outputs)[0]

        len_requires_grad_in_inputs = len(
            [t for t in flat_inputs if getattr(t, "requires_grad", False)]
        )
        len_requires_grad_in_outputs = len(
            [t for t in flat_outputs if getattr(t, "requires_grad", False)]
        )
        requires_grad_in_fn = isinstance(fn, nn.Module) and any(
            [p.requires_grad for p in fn.parameters()]
        )
        requires_grad = len_requires_grad_in_outputs > 0 and (
            requires_grad_in_fn or len_requires_grad_in_inputs > 0
        )
        forward_only = not (requires_grad and isinstance(outputs, torch.Tensor))

        # Latency benchmarking
        try:
            fwd_latencies, bwd_latencies = benchmark_node(
                self, forward_only=forward_only, device=device
            )
        except torch.cuda.OutOfMemoryError:
            fwd_latencies = np.array([MAX_VALUE])
            bwd_latencies = np.array([MAX_VALUE])
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

        logger.info(f"--- [PROFILE] node spec: {self}")
        logger.info(f"--- [PROFILE] detailed latency statistics:")
        logger.info(
            f"------ [PROFILE] fwd latency: [MEAN] {_format_time(fwd_latency_mean)}, [MEDIAN] {_format_time(fwd_latency_median)}, [STD] {fwd_latency_std:.8f}"
        )
        logger.info(
            f"------ [PROFILE] bwd latency: [MEAN] {_format_time(bwd_latency_mean)}, [MEDIAN] {_format_time(bwd_latency_median)}, [STD] {bwd_latency_std:.8f}"
        )

        return (fwd_latency_mean, fwd_latency_median, fwd_latency_std), (
            bwd_latency_mean,
            bwd_latency_median,
            bwd_latency_std,
        )

    @classmethod
    def from_callable(cls, fn: Callable, *args, **kwargs):
        # Because the target spec of a method would be a MethodDescriptorSpec,
        # we need to pass the object as the first argument to the bounded signature.
        if is_method(fn):
            args = (fn.__self__,) + args

        target_spec = TargetSpec.from_callable(fn)
        return cls(target_spec, *args, **kwargs)

    @classmethod
    def from_fx_node(cls, node: fx.Node, *args, **kwargs):
        if node.op == "call_method":
            target_spec = TargetSpec.from_fx_node(node, obj_for_method=args[0])
        else:
            target_spec = TargetSpec.from_fx_node(node)
        return cls(target_spec, *args, **kwargs)

    def _identity(self):
        return (self.target_spec, tuple(self.bounded_signature.arguments.values()))

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, NodeSpec):
            return False
        return self._identity() == __value._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def __repr__(self) -> str:
        return f"NodeSpec(target_spec={self.target_spec}, bounded_signature={dict(self.bounded_signature.arguments)})"


class TargetSpec:
    def __init__(self):
        pass

    @property
    def identifier(self):
        raise NotImplementedError

    @property
    def root_fn(self):
        raise NotImplementedError

    @property
    def signature(self):
        raise NotImplementedError

    def materialize(self, device="cuda"):
        raise NotImplementedError

    @staticmethod
    def from_callable(fn):
        fn = inspect.unwrap(fn)
        if isinstance(fn, nn.Module):
            return NNModuleSpec(fn)
        elif is_function(fn):
            return FunctionSpec(fn)
        elif is_methoddescriptor(fn):
            return MethodDescriptorSpec(fn)
        elif is_method(fn):
            obj = fn.__self__
            name = fn.__name__
            target = getattr(obj.__class__, name)
            unwrapped_target = inspect.unwrap(target)
            return MethodDescriptorSpec(unwrapped_target)
        else:
            raise ValueError(f"Cannot create a NodeSpec from {fn}")

    @classmethod
    def from_fx_node(cls, node, obj_for_method=None):
        graph = node.graph
        owning_module = graph.owning_module
        if node.op == "call_module":
            fn = getattr_recursive(owning_module, node.target)
        elif node.op == "call_function":
            fn = node.target
        elif node.op == "call_method":
            if obj_for_method is None:
                fn = getattr(torch.Tensor, node.target)
            else:
                fn = getattr(obj_for_method.__class__, node.target)
        else:
            raise ValueError(f"Unknown op {node.op}")
        return cls.from_callable(fn)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.identifier})"


class NNModuleSpec(TargetSpec):
    """
    NodeSpec for nn.Module

    A nn.Module is determined by its class, constants, and requires_grad
    """

    def __init__(self, module: nn.Module):
        super().__init__()

        if not isinstance(module, nn.Module):
            raise ValueError(f"Cannot create a NNModuleSpec from {module}")
        if is_module_symbolic(module):
            raise ValueError(
                f"Cannot create a NNModuleSpec for a symbolic module {module}"
            )

        self.op = "module"
        self.target = module.__class__
        self.constants = {
            name: getattr(module, name) for name in getattr(module, "__constants__", [])
        }
        self.param2requires_grad = {
            name: param.requires_grad for name, param in module.named_parameters()
        }
        self.param2dtype = {
            name: param.dtype for name, param in module.named_parameters()
        }

    @property
    def identifier(self):
        return f"{self.target.__module__}.{self.target.__name__} [constants] {self.constants}, [requires_grad] {self.param2requires_grad}, [dtype] {self.param2dtype}"

    @property
    def root_fn(self):
        return self.target.forward

    @property
    def signature(self):
        """
        Get the signature and remove the first argument (self) because it's bounded
        """
        signature = inspect.signature(self.root_fn)
        params = OrderedDict(signature.parameters)
        params.popitem(last=False)
        return inspect.Signature(params.values())

    def materialize(self, device="cuda"):
        module = self.target(**self.constants).to(device)
        for name, param in module.named_parameters():
            param.requires_grad = self.param2requires_grad[name]
        # Set the dtype of the parameters
        for name, param in module.named_parameters():
            param.data = param.data.to(self.param2dtype[name])
        return module


class FunctionSpec(TargetSpec):
    """
    NodeSpec for a function
    """

    def __init__(self, func):
        super().__init__()

        if not is_function(func):
            raise ValueError(f"{func} is not a function")

        self.op = "function"
        self.target = func

    @property
    def identifier(self):
        return f"{self.target.__module__}.{self.target.__name__}"

    @property
    def root_fn(self):
        return self.target

    @property
    def signature(self):
        """
        Normal torch.Tensor's methods do not have the signature so we need to
        inspect the signature of the overridden method
        """
        return inspect_torch_function_signature(self.root_fn)

    def materialize(self, device="cuda"):
        return self.root_fn

    def __reduce__(self):
        if self.identifier.startswith(("torch", "mist.modules")):
            return (self.from_identifier, (self.identifier,))
        else:
            return self.__class__, (self.target,)

    @classmethod
    def from_identifier(cls, identifier):
        fn = eval(identifier)
        unwrapped_fn = inspect.unwrap(fn)
        return cls(unwrapped_fn)


class MethodDescriptorSpec(TargetSpec):
    """
    NodeSpec for a method descriptor
    """

    def __init__(self, method_descriptor):
        super().__init__()

        if not is_methoddescriptor(method_descriptor):
            raise ValueError(f"{method_descriptor} is not a method descriptor")

        self.op = "method_descriptor"
        self.target = method_descriptor

    @property
    def identifier(self):
        obj_cls = self.target.__objclass__
        return f"{obj_cls.__module__}.{obj_cls.__name__}.{self.target.__name__}"

    @property
    def root_fn(self):
        return self.target

    @property
    def signature(self):
        """
        Normal torch.Tensor's methods do not have the signature so we need to
        inspect the signature of the overridden method
        """
        return inspect_torch_function_signature(self.root_fn)

    def materialize(self, device="cuda"):
        return self.root_fn


if __name__ == "__main__":
    # Test NNModuleSpec
    a = nn.Linear(4096, 4096)
    a_spec = TargetSpec.from_callable(a)
    a_input = torch.rand(4, 768, 4096)
    a_node_spec = NodeSpec.from_callable(a, a_input)

    # Test FunctionSpec
    b = torch.add
    b_spec = TargetSpec.from_callable(b)
    b_input = (torch.rand(10), torch.rand(10))
    b_node_spec = NodeSpec.from_callable(b, *b_input)

    # Test MethodDescriptorSpec
    c = torch.rand(10)
    c_spec = TargetSpec.from_callable(c.add)
    c_input = torch.rand(10)
    c_node_spec = NodeSpec.from_callable(c.add, c_input)

    # Benchmark
    a_node_spec.profile("cuda")
