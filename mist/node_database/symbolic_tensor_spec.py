from typing import Callable, Dict, List, Optional, Tuple, Union, Type, Any
from functools import partial

import sympy as sp
import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from mist import global_symbol_manager as gsm
from mist.node_database.tensor_spec import TensorSpec
from mist.sym_torch.symbolic_tensor import SymbolicTensor


class SymbolicTensorSpec:
    """
    A symbolic representation of a torch.Tensor.

    This class is used for the intermediate representation and would be later
    concretized to a TensorSpec by substituting the symbolic variables with
    concrete values.
    """

    def __init__(
        self,
        shape: Tuple[int],
        dtype: torch.dtype,
        requires_grad: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.device = device

    @classmethod
    def from_tensor(cls, tensor: SymbolicTensor):
        return cls(
            tuple(tensor.shape), tensor.dtype, tensor.requires_grad, tensor.device
        )

    def concretize(self, mapping):
        shape = gsm.subs(self.shape, mapping=mapping)
        return TensorSpec(shape, self.dtype, self.requires_grad, self.device)

    def __repr__(self) -> str:
        return f"SymbolicTensorSpec(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"


if __name__ == "__main__":
    b, s, z = gsm.symbols("b s z", (2, 3, 4))
    x = torch.rand(b, s, z)

    symbolic_tensor_spec = SymbolicTensorSpec.from_tensor(x)
    concrete_tensor_spec = symbolic_tensor_spec.concretize(gsm.mapping)
    materialized_tensor = concrete_tensor_spec.materialize()
