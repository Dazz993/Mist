import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op


@register_symbolic_op(torch, "layer_norm")
@register_symbolic_op(torch.nn.functional, "layer_norm")
class SymbolicLayerNorm(SymbolicOp):
    @staticmethod
    def apply(outputs, input, normalized_shape, weight=None, bias=None, eps=1e-5):
        pre_shape = input.shape[: -len(normalized_shape)]
        mean = torch.empty(pre_shape, dtype=input.dtype, device=input.device)
        invvar = torch.empty(pre_shape, dtype=input.dtype, device=input.device)
        ret = SymbolicTensor(outputs, symbolic_shape=input.shape)
        ret.context = SymbolicOpContext(
            op=SymbolicLayerNorm,
            saved_tensors=[input, weight, bias, mean, invvar],
        )
        return ret
