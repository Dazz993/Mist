import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op


@register_symbolic_op(torch, "clone")
@register_symbolic_op(torch.Tensor, "clone")
@register_symbolic_op(torch, "detach")
@register_symbolic_op(torch.Tensor, "detach")
@register_symbolic_op(torch.Tensor, "__deepcopy__")
@register_symbolic_op(torch.Tensor, "contiguous")
class SymbolicCloneDetach(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *args, **kwargs):
        return SymbolicTensor(outputs, input.shape)
