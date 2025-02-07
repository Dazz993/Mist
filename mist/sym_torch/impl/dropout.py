import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op


@register_symbolic_op(torch, "dropout")
@register_symbolic_op(torch.nn.functional, "dropout")
class SymbolicDropout(SymbolicOp):
    @staticmethod
    def apply(outputs, input, p=0.5, training=True, inplace=False):
        ret = SymbolicTensor(outputs, symbolic_shape=input.shape)

        if input.requires_grad and p != 0.0 and training:
            ret.context = SymbolicOpContext(
                op=SymbolicDropout,
                saved_tensors=[torch.empty_like(input, dtype=torch.bool)],
            )

        return ret
