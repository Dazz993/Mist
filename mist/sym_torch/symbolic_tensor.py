from __future__ import annotations
from typing import List, Tuple, Sequence, Optional, Any, Union, Callable

import numpy as np
import sympy as sp
import torch
from torch.types import _int
from torch.utils._pytree import tree_map

from mist.logger import get_logger
from mist.symbols import global_symbol_manager as gsm
from mist.sym_torch.symbolic_op import SymbolicOpContext
from mist.utils.tensor_entry import tensor_to_entry

logger = get_logger()

DEFAULT_DTYPE = torch.float32


def is_symbolic_complete(pytree):
    complete = True

    def fn(x):
        nonlocal complete
        if isinstance(x, SymbolicTensor):
            if not x.check_complete():
                complete = False

    tree_map(fn, pytree)
    return complete


def to_torch_tensor(pytree):
    def fn(x):
        if isinstance(x, SymbolicTensor):
            return x.to_torch_tensor()
        return x

    return tree_map(fn, pytree)


class SymbolicTensor(torch.Tensor):
    def __new__(
        cls,
        tensor: Union[torch.Tensor, SymbolicTensor],
        symbolic_shape: Tuple[int, sp.Basic],
    ):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"Expected tensor to be of type torch.Tensor, got {type(tensor)}"
            )

        # Map the symbolic shape to the tensor's shape in global symbol manager,
        # which will verify, check, or add a new mapping
        gsm.map(symbolic_shape, tensor.shape)

        # Create the symbolic tensor
        ret = tensor.as_subclass(cls)
        # ret._tensor = tensor
        ret._concrete_shape = (
            tuple(tensor.shape)
            if not isinstance(tensor, SymbolicTensor)
            else tensor.concrete_shape
        )
        ret._symbolic_shape = tuple(symbolic_shape)
        ret._context = None

        return ret

    def check_complete(self):
        return hasattr(self, "_concrete_shape") and hasattr(self, "_symbolic_shape")

    def to_torch_tensor(self):
        torch_tensor = self.as_subclass(torch.Tensor)
        if isinstance(self, torch.nn.Parameter) or getattr(self, "_is_param", False):
            torch_tensor.requires_grad_(self.requires_grad)
            torch_tensor._is_param = True
        return torch_tensor

    @property
    def shape(self):
        return self._symbolic_shape

    @property
    def concrete_shape(self):
        return self._concrete_shape

    @property
    def data(self):
        if hasattr(self, "_data"):
            return self._data
        return SymbolicTensor(self.to_torch_tensor().data, self.shape)

    @data.setter
    def data(self, value):
        self._data = value

    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def numel(self):
        if not self.shape:
            return 1
        return np.prod(self.shape)

    @property
    def _base_id(self):
        return id(self._base)

    @property
    def context(self) -> Optional[SymbolicOpContext]:
        return self._context

    @context.setter
    def context(self, context):
        if not isinstance(context, SymbolicOpContext):
            raise RuntimeError(
                f"Expected context to be of type SymbolicOpContext, got {type(context)}"
            )
        assert self._context is None, f"Context is already set to {self._context}"
        self._context = context

    def __repr__(self) -> str:
        return f"symbolic_tensor(..., shape={self.shape}, concrete_shape={self.concrete_shape}, dtype={self.dtype}, id(_base)={self._base_id})"

    """
    Note:
        The reason why we deprecated this for overwriting is because this can not cover all cases.
        ``__torch_function__`` will be called when subclass of torch.Tensor is used in torch ops.
        However, we may encounter that symbols being used in torch ops and thus resulting
        in a SymbolicTensor. In this case, __torch_function__ will not be called.

        However, it is now served as a check to ensure that the symbolic tensor is complete (op is supported).
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Overwrite this to support more functions for SymbolicTensor"""
        if kwargs is None:
            kwargs = {}

        output = super().__torch_function__(func, types, args, kwargs)
        if not is_symbolic_complete(output):
            if isinstance(output, torch.Tensor):
                output_info = tensor_to_entry(output.to_torch_tensor())
            else:
                output_info = tuple(
                    tensor_to_entry(o.to_torch_tensor()) for o in output
                )
            raise RuntimeError(
                f"Output is not symbolic complete. "
                f"This could happen when this op is not overriden for symbolic tensors.\n"
                f"    - Func: {func}\n"
                f"    - Args: {args}\n"
                f"    - Kwargs: {kwargs}\n"
                f"    - Output: {output_info}\n"
            )
        return output


if __name__ == "__main__":
    from mist import global_symbol_manager as gsm

    b = gsm.symbols("b", 10, integer=True, positive=True)
    x = torch.empty((b, 5), device="meta")
    a = x.log()
