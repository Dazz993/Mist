from typing import Callable, Dict, List, Optional, Tuple, Union, Type, Any
from functools import partial

import sympy as sp
import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from mist.utils.initialization import init_empty_weights
from mist.utils.memory import materialize_module, materialize_tensor
from mist.sym_torch.symbolic_tensor import SymbolicTensor
from mist.tracer.hf import _MANUAL_META_OVERRIDES


def is_symbolic_shape(shape):
    if isinstance(shape, (tuple, list)):
        return any(isinstance(x, sp.Basic) for x in shape)
    return isinstance(shape, sp.Basic)


class TensorSpec:
    """
    A class to represent a concrete torch.Tensor.
    """

    keys = ["shape", "dtype", "requires_grad"]

    def __init__(
        self,
        shape: Tuple[int],
        dtype: torch.dtype,
        requires_grad: bool = False,
        device: Optional[torch.device] = None,
    ):
        if is_symbolic_shape(shape):
            raise RuntimeError(
                f"Expected shape to be a concrete tuple, got {shape} of type {type(shape)}"
            )

        self.shape = shape
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.device = device

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"Expected tensor to be of type torch.Tensor, got {type(tensor)}"
            )
        if isinstance(tensor, SymbolicTensor):
            raise RuntimeError("Cannot create a TensorSpec from a SymbolicTensor.")
        return cls(tuple(tensor.shape), tensor.dtype, tensor.requires_grad)

    def materialize(self, device="cuda", rand=True):
        return materialize_tensor(self, device=device, rand=rand)

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
