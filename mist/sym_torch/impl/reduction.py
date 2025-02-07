import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op


@register_symbolic_op(torch, "sum")
@register_symbolic_op(torch.Tensor, "sum")
@register_symbolic_op(torch, "mean")
@register_symbolic_op(torch.Tensor, "mean")
class SymbolicReduction(SymbolicOp):
    @staticmethod
    def apply(outputs, input, dim=(), keepdim=False, dtype=None):
        if not dim:
            dim = tuple(range(input.ndim))
        if not isinstance(dim, (tuple, list)):
            dim = (dim,)
        dim = tuple(d + input.ndim if d < 0 else d for d in dim)
        if keepdim:
            symbolic_shape = tuple(
                input.shape[i] if i not in dim else 1 for i in range(input.ndim)
            )
        else:
            symbolic_shape = tuple(
                input.shape[i] for i in range(input.ndim) if i not in dim
            )

        ret = SymbolicTensor(outputs, symbolic_shape=symbolic_shape)
        ret.context = SymbolicOpContext(
            op=SymbolicReduction,
            saved_tensors=[input],
        )

        return ret
