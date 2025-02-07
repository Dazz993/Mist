from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union, Type, Any
import torch
from torch import fx, nn
from torch.utils._pytree import tree_map
import inspect
import operator
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from copy import deepcopy

import sympy as sp
import torch.nn as nn

from mist import global_symbol_manager as gsm
from mist.distributed.overrides import MistProcessGroup
from mist.node_database.tensor_spec import (
    TensorSpec,
)
from mist.node_database.symbolic_tensor_spec import (
    SymbolicTensorSpec,
)
from mist.node_database.node_spec import (
    NodeSpec,
    TargetSpec,
    NNModuleSpec,
    MethodDescriptorSpec,
    FunctionSpec,
)
from mist.utils.module import getattr_recursive
from mist.utils.initialization import init_empty_weights
from mist.utils.memory import materialize_module, materialize_tensor
from mist.sym_torch.symbolic_tensor import SymbolicTensor
from mist.tracer.hf import _MANUAL_META_OVERRIDES
from mist.utils.tensor_entry import TensorEntry
from mist.node_database.node_spec import is_function, is_method


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


def map_to_symbolic_tensor_spec(x):
    if isinstance(x, (torch.Tensor, TensorEntry)):
        return SymbolicTensorSpec.from_tensor(x)
    return x


def map_to_materialized_tensor(x, device="meta", rand=True):
    if isinstance(x, TensorSpec):
        return x.materialize(device, rand=rand)
    return x


def map_to_concrete_tensor_spec(x, mapping):
    if isinstance(x, SymbolicTensorSpec):
        return x.concretize(mapping)
    elif isinstance(x, sp.Basic):
        return gsm.subs(x, mapping=mapping)
    elif isinstance(x, MistProcessGroup):
        return MistProcessGroup(
            world_size=gsm.subs(x._world_size, mapping=mapping),
            rank=gsm.subs(x._rank, mapping=mapping),
            global_rank=gsm.subs(x._global_rank, mapping=mapping),
        )
    return x


# def is_function(fn):
#     return inspect.isfunction(fn) or (
#         inspect.isbuiltin(fn)
#         and (fn.__self__ is None or fn.__module__ in ("_operator", "torch._C._nn"))
#     )


# def is_method(fn):
#     return inspect.ismethod(fn) or (
#         inspect.isbuiltin(fn)
#         and (fn.__self__ is not None and fn.__module__ in ("_operator", "torch._C._nn"))
#     )


def is_methoddescriptor(fn):
    return inspect.ismethoddescriptor(fn)


class SymbolicNodeSpec:
    def __init__(self, target_spec, *args, **kwargs):
        self.target_spec = target_spec

        args_spec = tree_map(map_to_symbolic_tensor_spec, args)
        kwargs_spec = tree_map(map_to_symbolic_tensor_spec, kwargs)
        # print(f"{target_spec=}")
        # print(f"{target_spec.signature=}")
        # print(f"{args_spec=}")
        # print(f"{kwargs_spec=}")
        bounded_signature = target_spec.signature.bind(*args_spec, **kwargs_spec)
        bounded_signature.apply_defaults()
        # Bounded signature is used for identifying the node.
        self.bounded_signature = bounded_signature
        # Args and kwargs are used for concretizing the node.
        self.args = args_spec
        self.kwargs = kwargs_spec

    def concretize(self, mapping):
        if isinstance(self.target_spec, SymbolicNNModuleSpec):
            concrete_target_spec = self.target_spec.concretize(mapping)
        else:
            concrete_target_spec = self.target_spec

        _map_to_concrete_tensor_spec = partial(
            map_to_concrete_tensor_spec, mapping=mapping
        )
        args = tree_map(_map_to_concrete_tensor_spec, self.args)
        kwargs = tree_map(_map_to_concrete_tensor_spec, self.kwargs)
        return NodeSpec(concrete_target_spec, *args, **kwargs)

    def symbols(self):
        memo = set()

        def _fn_collect_variable(x):
            if isinstance(x, sp.Basic):
                for s in x.free_symbols:
                    memo.add(s)
            if isinstance(x, (SymbolicTensorSpec, SymbolicTensor)):
                tree_map(_fn_collect_variable, x.shape)

        if isinstance(self.target_spec, SymbolicNNModuleSpec):
            tree_map(_fn_collect_variable, self.target_spec.constants)
        tree_map(_fn_collect_variable, self.args)
        tree_map(_fn_collect_variable, self.kwargs)

        # Check the params and buffers of the module if the target is a module.
        if isinstance(self.target_spec, SymbolicNNModuleSpec):
            ori_memo = deepcopy(memo)
            tree_map(_fn_collect_variable, list(self.target_spec._module.parameters()))
            tree_map(_fn_collect_variable, list(self.target_spec._module.buffers()))
            if len(memo) != len(ori_memo):
                raise ValueError(
                    f"The parameters or buffers of the module cannot be symbolic. {memo - ori_memo}"
                )

        return memo

    @classmethod
    def from_callable(cls, fn: Callable, *args, **kwargs):
        # Because the target spec of a method would be a MethodDescriptorSpec,
        # we need to pass the object as the first argument to the bounded signature.
        if is_method(fn):
            args = (fn.__self__,) + args

        target_spec = SymbolicTargetSpec.from_callable(fn)
        return cls(target_spec, *args, **kwargs)

    @classmethod
    def from_fx_node(cls, node: fx.Node, *args, **kwargs):
        if node.op == "call_method":
            target_spec = SymbolicTargetSpec.from_fx_node(node, obj_for_method=args[0])
        else:
            target_spec = SymbolicTargetSpec.from_fx_node(node)
        return cls(target_spec, *args, **kwargs)

    def _identity(self):
        return (self.target_spec, self.bounded_signature)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, NodeSpec):
            return False
        return self._identity() == __value._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def __repr__(self) -> str:
        return f"NodeSpec(target_spec={self.target_spec}, bounded_signature={self.bounded_signature})"


class SymbolicTargetSpec:
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

    def concretize(self, mapping):
        raise NotImplementedError

    @staticmethod
    def from_callable(fn):
        fn = inspect.unwrap(fn)
        if isinstance(fn, nn.Module):
            return SymbolicNNModuleSpec(fn)
        elif is_function(fn):
            return FunctionSpec(fn)
        elif is_methoddescriptor(fn):
            return MethodDescriptorSpec(fn)
        elif is_method(fn):
            obj = fn.__self__
            obj_class = obj.__class__
            name = fn.__name__
            target = getattr(obj_class, name)
            target = inspect.unwrap(target)
            return MethodDescriptorSpec(target)
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
            if obj_for_method is None or isinstance(obj_for_method, TensorEntry):
                fn = getattr(torch.Tensor, node.target)
            else:
                fn = getattr(obj_for_method.__class__, node.target)
        else:
            raise ValueError(f"Unknown op {node.op}")
        return cls.from_callable(fn)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.identifier})"


class SymbolicNNModuleSpec(SymbolicTargetSpec):
    """
    SymbolicNodeSpec for nn.Module

    A nn.Module is determined by its class, constants, and requires_grad
    """

    def __init__(self, module: nn.Module):
        super().__init__()

        if not isinstance(module, nn.Module):
            raise ValueError(f"Cannot create a NNModuleSpec from {module}")

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

        # self._module = module

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

    def concretize(self, mapping):
        concrete_constants = gsm.subs(self.constants, mapping=mapping)
        with init_empty_weights():
            module = self.target(**concrete_constants)
            for name, param in module.named_parameters():
                param.requires_grad = self.param2requires_grad[name]
            # Set the dtype of the parameters
            for name, param in module.named_parameters():
                param.data = param.data.to(self.param2dtype[name])
        return NNModuleSpec(module)


if __name__ == "__main__":
    from mist import global_symbol_manager as gsm

    b, s, z = gsm.symbols("b s z", (2, 3, 4))
    input = torch.rand(b, s, 128)
    other = torch.rand(b, s, 128)

    h = gsm.symbols("h", 5)

    # Test NNModuleSpec
    nn.Linear.reset_parameters = lambda self: None
    a = nn.Linear(128, z)
    a_node_spec = SymbolicNodeSpec.from_callable(a, input)
    a_concrete_node_spec = a_node_spec.concretize(gsm.mapping)

    # Should raise an error because `h` is added
    # d = nn.Linear(128, z)
    # d.weight = torch.nn.Parameter(torch.rand(128, h))
    # d_node_spec = SymbolicNodeSpec.from_callable(d, input)
    # d_concrete_node_spec = d_node_spec.concretize(gsm.mapping)
    # print(d_node_spec.variables())

    # Test FunctionSpec
    b = torch.add
    b_node_spec = SymbolicNodeSpec.from_callable(b, input, other)
    b_concrete_node_spec = b_node_spec.concretize(gsm.mapping)

    # Test MethodDescriptorSpec
    c_node_spec = SymbolicNodeSpec.from_callable(input.add, other)
    c_concrete_node_spec = c_node_spec.concretize(gsm.mapping)
