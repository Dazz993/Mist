import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op


@register_symbolic_op(torch, "embedding")
@register_symbolic_op(torch.nn.functional, "embedding")
class SymbolicEmbedding(SymbolicOp):
    @staticmethod
    def apply(outputs, input, weight, *args, **kwargs):
        ret = SymbolicTensor(
            outputs,
            symbolic_shape=input.shape + weight.shape[1:],
        )

        saved_tensors = []
        if weight.requires_grad:
            saved_tensors.append(input)
        ret.context = SymbolicOpContext(
            op=SymbolicEmbedding,
            saved_tensors=saved_tensors,
        )
        return ret
