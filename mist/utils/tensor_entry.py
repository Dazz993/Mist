from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import sympy as sp
import torch
from torch.utils._pytree import tree_map


@dataclass
class TensorEntry:
    cls: type
    shape: Tuple[Union[int, sp.Basic]]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool
    id: int
    base_id: int
    comment: str = ""

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, comment: str = ""):
        return cls(
            cls=type(tensor),
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
            id=id(tensor),
            base_id=id(tensor) if tensor._base is None else id(tensor._base),
            comment=comment,
        )

    def copy(self):
        return TensorEntry(
            cls=self.cls,
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            id=self.id,
            base_id=self.base_id,
            comment=self.comment,
        )

    def __repr__(self) -> str:
        comment_str = f', comment="{self.comment}"' if self.comment else ""
        return (
            f"{self.cls.__name__}["
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            # f"device={self.device}, "
            f"requires_grad={self.requires_grad}, "
            f"base_id={self.base_id}"
            f"{comment_str}]"
        )

    def nbytes(self) -> int:
        return np.prod(self.shape) * torch.empty([], dtype=self.dtype).element_size()

    def _identity(self):
        return (self.base_id,)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorEntry):
            return False
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())


def tensor_to_entry(tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return TensorEntry.from_tensor(tensor)


def tree_to_entries(tree):
    return tree_map(tensor_to_entry, tree)


def tree_to_ids(tree):
    return tree_map(id, tree)


def get_tensor_base_id(tensor):
    if not isinstance(tensor, torch.Tensor):
        return id(tensor)
    return id(tensor) if tensor._base is None else id(tensor._base)
