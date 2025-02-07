import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op
from mist.sym_torch.impl._shape_prop import get_add_shape


@register_symbolic_op(torch.nn.functional, "scaled_dot_product_attention")
class SymbolicScaledDotProductAttention(SymbolicOp):
    """
    Scale Dot Product Attention

    See ``https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html`` for details.
    """

    @staticmethod
    def apply(outputs, query, key, value, *args, **kwargs):
        if len(args) != 0:
            attn_mask = args[0]
        else:
            attn_mask = kwargs.pop("attn_mask", None)

        pre_shape_q = query.shape[:-2]
        pre_shape_k = key.shape[:-2]
        pre_shape_v = value.shape[:-2]
        assert (
            len(pre_shape_q) == len(pre_shape_k) == len(pre_shape_v)
        ), f"query, key, value must have same number of dimensions, got {len(pre_shape_q)}, {len(pre_shape_k)}, {len(pre_shape_v)}"
        pre_shape = get_add_shape(get_add_shape(pre_shape_q, pre_shape_k), pre_shape_v)

        l, e = query.shape[-2:]
        s, e_ = key.shape[-2:]
        s_, ev = value.shape[-2:]
        assert (
            e == e_
        ), f"query and key must have same embedding dimension, got {e} and {e_}"
        assert (
            s == s_
        ), f"key and value must have same sequence length, got {s} and {s_}"
        post_shape = (l, ev)

        ret = SymbolicTensor(outputs, symbolic_shape=pre_shape + post_shape)
        saved_tensors = [None, None, None]
        if query.requires_grad:
            saved_tensors[1] = query
            saved_tensors[2] = value
        if key.requires_grad:
            saved_tensors[0] = key
            saved_tensors[2] = value
        if value.requires_grad:
            saved_tensors[0] = key
            saved_tensors[1] = query
        ret.context = SymbolicOpContext(
            op=SymbolicScaledDotProductAttention, saved_tensors=saved_tensors
        )

        return ret
