import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op


@register_symbolic_op(torch, "outer")
@register_symbolic_op(torch.Tensor, "outer")
class SymbolicProduct(SymbolicOp):
    @staticmethod
    def apply(outputs, input, vec2, **kwargs):
        shape = input.shape + vec2.shape
        ret = SymbolicTensor(outputs, symbolic_shape=shape)
        ret.context = SymbolicOpContext(
            op=SymbolicProduct,
            saved_tensors=[input, vec2],
        )
        return ret
