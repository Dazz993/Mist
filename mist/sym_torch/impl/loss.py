import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op


@register_symbolic_op(torch.nn.functional, "cross_entropy")
@register_symbolic_op(torch._C._nn, "cross_entropy_loss")
class SymbolicCrossEntropy(SymbolicOp):
    @staticmethod
    def apply(outputs, input, target, weight=None, **kwargs):
        ret = SymbolicTensor(outputs, symbolic_shape=())
        if input.requires_grad:
            ret.context = SymbolicOpContext(
                op=SymbolicCrossEntropy,
                saved_tensors=[input, torch.empty_like(input, requires_grad=True)],
            )
        return ret
