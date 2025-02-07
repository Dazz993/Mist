from typing import Callable, Dict, List, Optional, Tuple, Union, Type, Any
import torch
from torch import fx, nn
from torch.utils._pytree import tree_map
import inspect
import operator
from collections import OrderedDict
from dataclasses import dataclass, field
from copy import deepcopy

from mist.sym_torch.symbolic_tensor import SymbolicTensor
from mist.utils.memory import materialize_module, materialize_tensor


def map_to_tensor_spec(x):
    if isinstance(x, torch.Tensor):
        return TensorSpec.from_tensor(x)
    return x


def map_to_materialized_tensor(x, device="meta"):
    if isinstance(x, TensorSpec):
        return x.instantiate(device)
    return x


class TensorSpec:
    keys = ["shape", "dtype", "requires_grad"]

    def __init__(
        self, shape: Tuple[int], dtype: torch.dtype, requires_grad: bool = False
    ):
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = requires_grad

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"Expected tensor to be of type torch.Tensor, got {type(tensor)}"
            )
        if isinstance(tensor, SymbolicTensor):
            raise RuntimeError("Cannot create a TensorSpec from a SymbolicTensor.")
        return cls(tuple(tensor.shape), tensor.dtype, tensor.requires_grad)

    def materialize(self, device, rand=True):
        return materialize_tensor(self, device, rand=rand)

    def __repr__(self) -> str:
        return f"TensorSpec(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def _identity(self):
        return (getattr(self, key) for key in self.keys)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, TensorSpec):
            return False
        return self._identity() == __value._identity()

    def __hash__(self) -> int:
        return hash(self._identity())


class InputsSpec:
    def __init__(self, signature: inspect.Signature, *args, **kwargs):
        # Map all tensors in args and kwargs to TensorSpec
        self.args = tree_map(map_to_tensor_spec, args)
        self.kwargs = tree_map(map_to_tensor_spec, kwargs)

        _signature = deepcopy(signature)
        _signature.bind(*args, **kwargs)
        _signature.apply_defaults()
        self.bounded_signature = _signature

    def _identity(self):
        return (self.bounded_signature,)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, InputsSpec):
            return False
        return self._identity() == __value._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def __repr__(self) -> str:
        return f"InputsSpec({self.bounded_signature})"


class OutputSpec:
    def __init__(self, output: Any):
        self.output = tree_map(map_to_tensor_spec, output)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, OutputSpec):
            return False
        return self.output == __value.output

    def __hash__(self) -> int:
        return hash(self.output)

    def __repr__(self) -> str:
        return f"OutputSpec({self.output})"


class UndeterminedOutput:
    def __repr__(self) -> str:
        return f"UndeterminedOutput()"


class UndeterminedOutputSpec(OutputSpec):
    def __init__(self):
        super().__init__(UndeterminedOutput())


class EmptyOutput:
    def __repr__(self) -> str:
        return f"EmptyOutput()"


class EmptyOutputSpec(OutputSpec):
    def __init__(self):
        super().__init__(EmptyOutput())
