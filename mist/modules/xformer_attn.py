import math
import torch
import torch.fx
import torch.nn as nn
import xformers.ops as xops

from mist import global_symbol_manager as gsm
from mist.overrides import register_overriden_func
from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op
from mist.modules.sym_register import map_to_symbolic_tensors


# Function Interface for tracing
def xformer_memory_efficient_attention(
    qkv, causal=True, key_padding_mask=None, dropout=0.0, softmax_scale=None
):
    """Implements the multihead softmax attention.
    Arguments
    ---------
        qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
        causal: if passed, will override self.causal
        key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
            False means to mask out. (B, S)
    """
    q, k, v = qkv.unbind(dim=2)
    batch_size, seqlen, num_heads, head_dim = q.shape
    softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])

    # Compute the bias
    attn_bias = None
    if key_padding_mask is not None:
        padding_mask = torch.full(
            (batch_size, seqlen), -10000.0, dtype=qkv.dtype, device=qkv.device
        )
        padding_mask.masked_fill_(key_padding_mask, 0.0)
        attn_bias = padding_mask.reshape(batch_size, 1, 1, seqlen)
        # Expand the mask to the same shape of q
        attn_bias = attn_bias.expand(-1, num_heads, seqlen, -1)

    if causal and attn_bias is None:
        attn_bias = xops.LowerTriangularMask()
    elif causal:
        # ===============================================================================
        # Only works with FwOp
        # attn_bias = xops.fmha.attn_bias.LowerTriangularMaskWithTensorBias(attn_bias)
        # ===============================================================================
        # Compatibility for both FwOp and BwOp
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=qkv.device, dtype=qkv.dtype),
            1,
        )
        attn_bias = attn_bias + causal_mask

    # Compute the attention
    output = xops.memory_efficient_attention(
        query=q,
        key=k,
        value=v,
        attn_bias=attn_bias,
        p=dropout,
        scale=softmax_scale,
        op=(xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp),
    )

    return output


torch.fx.wrap("xformer_memory_efficient_attention")


# Symbolic Support
@register_symbolic_op(torch.ops.xformers, "efficient_attention_forward_cutlass")
@register_symbolic_op(xops.fmha.cutlass.FwOp, "OPERATOR")
class SymbolicXformerMemoryEfficientAttentionForwardCutlass(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        query,
        key,
        value,
        attn_bias,
        *args,
        **kwargs,
    ):
        out, lse, rng_seed, rng_offset = outputs
        b, s, h, d = query.shape
        sym_out = SymbolicTensor(out, (b, s, h, d))
        # Omit the LSE for now
        # sym_lse = SymbolicTensor(lse, (b, h, s))
        sym_lse = lse

        return (
            sym_out,
            sym_lse,
            rng_seed,
            rng_offset,
        )


@register_symbolic_op(torch.ops.xformers, "efficient_attention_backward_cutlass")
@register_symbolic_op(xops.fmha.cutlass.BwOp, "OPERATOR")
class SymbolicXformerMemoryEfficientAttentionBackwardCutlass(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        grad,
        query,
        key,
        value,
        attn_bias,
        *args,
        **kwargs,
    ):
        grad_q, grad_k, grad_v, grad_bias = outputs
        b, s, h, d = query.shape
        symbolic_shapes = [
            (b, s, h, d),
            (b, s, h, d),
            (b, s, h, d),
            (b, h, s, s),
        ]
        ret = map_to_symbolic_tensors(outputs, symbolic_shapes)
        print(ret)
        return ret


# Disable the check for the operator support
@register_overriden_func(xops.fmha.cutlass.FwOp, "not_supported_reasons")
def dummy_xops_fmha_cutlass_fwop_not_supported_reasons(*args, **kwargs):
    return []


@register_overriden_func(xops.fmha.cutlass.BwOp, "not_supported_reasons")
def dummy_xops_fmha_cutlass_bwop_not_supported_reasons(*args, **kwargs):
    return []


# Support Symbolic Tensor Type for ATTN BIAS
xops.fmha.cutlass.FwOp.SUPPORTED_ATTN_BIAS_TYPES.add(SymbolicTensor)
xops.fmha.cutlass.BwOp.SUPPORTED_ATTN_BIAS_TYPES.add(SymbolicTensor)


class XFormersSelfAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout

    def forward(self, qkv, causal=None, key_padding_mask=None):
        return xformer_memory_efficient_attention(
            qkv,
            causal=causal or self.causal,
            key_padding_mask=key_padding_mask,
            dropout=self.attention_dropout,
            softmax_scale=self.softmax_scale,
        )


if __name__ == "__main__":
    attn_layer = XFormersSelfAttention(causal=True)

    causal = True

    b, s, h, d = gsm.symbols("b s h d", (2, 2048, 32, 128))
    tp = gsm.symbols("tp", (2,))
    # qkv = torch.randn(b, s, 3, h, d, device="cuda", requires_grad=True)
    # qkv.retain_grad()

    # key_padding_mask = torch.zeros(b, s, device="cuda").bool()
    key_padding_mask = None

    Wqkv = torch.nn.Linear(h * d, 3 * h * d / tp, bias=False, device="cuda")

    input_tensor = torch.randn(b, s, h * d, device="cuda", requires_grad=True)
    qkv = Wqkv(input_tensor).view(b, s, 3, h, d)

    qkv.sum().backward()

    output = attn_layer(qkv, causal=causal, key_padding_mask=key_padding_mask)

    torch.autograd.backward(
        output, grad_tensors=torch.ones_like(output), retain_graph=True
    )
