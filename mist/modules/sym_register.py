import sympy as sp
import torch
from typing import Optional, Union

import numpy as np
import mist
from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import (
    register_symbolic_op,
    map_symbols_to_concretes_single,
)
from mist.modules.triton import rotary
from mist.utils.sympy import fake_floordiv
from mist.utils.tensor_entry import tree_to_entries
from mist.modules import losses
from mist.utils.memory import make_viewless_tensor


from mist.modules import activations
import dropout_layer_norm
import flash_attn_2_cuda as flash_attn_cuda
import fused_dense_lib as fused_dense_cuda

# apex/fused_softmax
try:
    import scaled_upper_triang_masked_softmax_cuda
except ImportError:
    scaled_upper_triang_masked_softmax_cuda = None

try:
    import scaled_masked_softmax_cuda
except ImportError:
    scaled_masked_softmax_cuda = None


def map_to_symbolic_tensors(outputs, sym_shapes):
    assert len(outputs) == len(sym_shapes), (
        f"outputs and sym_shapes should have the same length, "
        f"got {len(outputs)} and {len(sym_shapes)}"
    )
    ret = []
    for out, shape in zip(outputs, sym_shapes):
        if (
            isinstance(out, torch.Tensor)
            and shape is not None
            and any(isinstance(s, sp.Basic) for s in shape)
        ):
            ret.append(SymbolicTensor(out, shape))
        else:
            ret.append(out)
    return ret


@register_symbolic_op(rotary, "apply_rotary")
class SymbolicApplyRotary(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        interleaved=False,
        inplace=False,
        conjugate=False,
    ):
        return SymbolicTensor(outputs, x.shape)


@register_symbolic_op(flash_attn_cuda, "fwd")
class SymbolicFalshAttnFwd(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
        *args,
        **kwargs,
    ):
        # outputs: out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state
        # ----------------
        # q:     (B, S_q, H, D)
        # k:     (B, S_k, H, D)
        # v:     (B, S_k, H, D)
        # out:   (B, S_q, H, D)       | None

        b, s_q, h, d = q.shape
        _, s_k, _, _ = k.shape

        sym_out_shape = (b, s_q, h, d)
        sym_q_shape = q.shape
        sym_k_shape = k.shape
        sym_v_shape = v.shape
        sym_out_padded_shape = (b, s_q, h, d)
        sym_softmax_lse_shape = (b, h, s_q)
        sym_S_dmask_shape = (b, h, s_q, s_k)
        sym_shapes = [
            sym_out_shape,
            sym_q_shape,
            sym_k_shape,
            sym_v_shape,
            sym_out_padded_shape,
            sym_softmax_lse_shape,
            sym_S_dmask_shape,
            None,  # rng_state
        ]

        return map_to_symbolic_tensors(outputs, sym_shapes)


@register_symbolic_op(flash_attn_cuda, "varlen_fwd")
class SymbolicFlashAttnVarlenFwd(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        *args,
        **kwargs,
        # p_dropout: float,
        # softmax_scale: float,
        # zero_tensors: bool,
        # is_causal: bool,
        # window_size_left: int,
        # window_size_right: int,
        # return_softmax: bool,
        # gen: Optional[torch.Generator],
    ):
        # outputs: out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state,
        # ----------------
        # q:            (total_q, H, D)
        # k:            (total_k, H, D)
        # v:            (total_k, H, D)
        # out:          (total_q, H, D)       | None
        # cu_seqlens_q: (B + 1,)
        # cu_seqlens_k: (B + 1,)

        total_q, h, d = q.shape
        total_k, _, _ = k.shape
        b = cu_seqlens_q.shape[0] - 1

        sym_out_shape = (total_q, h, d)
        sym_q_shape = q.shape
        sym_k_shape = k.shape
        sym_v_shape = v.shape
        sym_out_padded_shape = (total_q, h, d)
        sym_softmax_lse_shape = (b, h, max_seqlen_q)
        sym_S_dmask_shape = (b, h, max_seqlen_q, max_seqlen_k)
        sym_shapes = [
            sym_out_shape,
            sym_q_shape,
            sym_k_shape,
            sym_v_shape,
            sym_out_padded_shape,
            sym_softmax_lse_shape,
            sym_S_dmask_shape,
            None,  # rng_state
        ]

        return map_to_symbolic_tensors(outputs, sym_shapes)


@register_symbolic_op(flash_attn_cuda, "bwd")
class SymbolicFlashAttnBwd(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
        softmax_lse: Optional[torch.Tensor],
        dq: Optional[torch.Tensor],
        dk: Optional[torch.Tensor],
        dv: Optional[torch.Tensor],
        *args,
        **kwargs,
    ):
        # outputs: dq, dk, dv, softmax_d
        # ----------------
        # dout:  (B, S_q, H, D)
        # q:     (B, S_q, H, D)
        # k:     (B, S_k, H, D)
        # v:     (B, S_k, H, D)
        # out:   (B, S_q, H, D)       | None
        # dq:    (B, S_q, H, D)       | None
        # dk:    (B, S_k, H, D)       | None
        # dv:    (B, S_k, H, D)       | None

        b, s_q, h, d = dout.shape
        _, s_k, _, _ = k.shape

        sym_dq_shape = (b, s_q, h, d)
        sym_dk_shape = (b, s_k, h, d)
        sym_dv_shape = (b, s_k, h, d)
        sym_softmax_d_shape = (b, h, s_q)
        sym_shapes = [
            sym_dq_shape,
            sym_dk_shape,
            sym_dv_shape,
            sym_softmax_d_shape,
        ]

        return map_to_symbolic_tensors(outputs, sym_shapes)


@register_symbolic_op(flash_attn_cuda, "varlen_bwd")
class SymbolicFlashAttnVarlenBwd(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
        softmax_lse: Optional[torch.Tensor],
        dq: Optional[torch.Tensor],
        dk: Optional[torch.Tensor],
        dv: Optional[torch.Tensor],
        cu_seq_lens_q: Optional[torch.Tensor],
        cu_seq_lens_k: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        *args,
        **kwargs,
    ):
        # outputs: dq, dk, dv, softmax_d
        # ----------------
        # dout:             (total_q, H, D)
        # q:                (total_q, H, D)
        # k:                (total_k, H, D)
        # v:                (total_k, H, D)
        # out:              (total_q, H, D)       | None
        # dq:               (total_q, H, D)       | None
        # dk:               (total_k, H, D)       | None
        # dv:               (total_k, H, D)       | None
        # cu_seq_lens_q:    (B + 1,)
        # cu_seq_lens_k:    (B + 1,)

        total_q, h, d = dout.shape
        total_k, _, _ = k.shape
        b = cu_seq_lens_q.shape[0] - 1

        sym_dq_shape = (total_q, h, d)
        sym_dk_shape = (total_k, h, d)
        sym_dv_shape = (total_k, h, d)
        sym_softmax_d_shape = (b, h, max_seqlen_q)
        sym_shapes = [
            sym_dq_shape,
            sym_dk_shape,
            sym_dv_shape,
            sym_softmax_d_shape,
        ]

        return map_to_symbolic_tensors(outputs, sym_shapes)


@register_symbolic_op(fused_dense_cuda, "linear_bias_wgrad")
class SymbolicFusedDenseLinearBiasWGrad(SymbolicOp):
    @staticmethod
    def apply(outputs, input: torch.Tensor, d_output: torch.Tensor, has_d_bias: bool):
        # outputs: d_weight, d_bias
        # ----------------
        # input:     (B, I)
        # d_output:  (B, O)

        b, i = input.shape
        _, o = d_output.shape

        sym_d_weight_shape = (o, i)
        sym_d_bias_shape = (o,)
        sym_shapes = [
            sym_d_weight_shape,
            sym_d_bias_shape,
        ]

        return map_to_symbolic_tensors(outputs, sym_shapes)


@register_symbolic_op(fused_dense_cuda, "linear_act_forward")
class SymbolicFusedDenseLinearActForward(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        is_gelu: bool,
        save_pre_act: bool,
        heuristic: int,
    ):
        # outputs: output, pre_act
        # ----------------
        # input:     (B, I)
        # weight:    (O, I)
        # bias:      (O,)
        # output:    (B, O)
        # pre_act:   (B, O) or (B, O / 8) if is_gelu

        b, i = input.shape
        o, _ = weight.shape

        sym_output_shape = (b, o)
        sym_pre_act_shape = (b, o / 8) if is_gelu else (b, o)
        sym_shapes = [
            sym_output_shape,
            sym_pre_act_shape,
        ]

        return map_to_symbolic_tensors(outputs, sym_shapes)


@register_symbolic_op(fused_dense_cuda, "bias_act_linear_dgrad_bgrad")
class SymbolicFusedDenseBiasActLinearDgradBgrad(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        weight: torch.Tensor,
        d_output: torch.Tensor,
        pre_act: torch.Tensor,
        is_gelu: bool,
        heuristic: int,
    ):
        # outputs: d_input, d_bias
        # ----------------
        # weight:    (O, I)
        # d_output:  (B, O)
        # pre_act:   (B, O) or (B, O / 8) if is_gelu

        b, o = d_output.shape
        _, i = weight.shape

        sym_d_input_shape = (b, i)
        sym_d_bias_shape = (i,)
        sym_shapes = [
            sym_d_input_shape,
            sym_d_bias_shape,
        ]

        return map_to_symbolic_tensors(outputs, sym_shapes)


@register_symbolic_op(dropout_layer_norm, "dropout_add_ln_fwd")
class SymbolicDropoutAddLNFwd(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        x0: torch.Tensor,
        residual: Optional[torch.Tensor],
        gamma: torch.Tensor,
        beta: Optional[torch.Tensor],
        rowscale: Optional[torch.Tensor],
        colscale: Optional[torch.Tensor],
        x0_subset: Optional[torch.Tensor],
        z_subset: Optional[torch.Tensor],
        dropout_p: float,
        eps: float,
        rowscale_const: float,
        z_numrows: int,
        gen_,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
    ):
        # outputs: zmat, xmat, dmask, mu, rsigma
        # ----------------
        # x0:           (B * S, H)
        # residual:     (B * S, H) | None
        # gamma:        (H,)
        # beta:         (H,)       | None
        # rowscale:     (B * S,)   | None
        # colscale:     (H,)       | None
        # x0_subset:    (B * S)    | None
        # z_subset:     (B * S)    | None

        rows = x0_subset.shape[0] if x0_subset is not None else x0.shape[0]
        cols = x0.shape[1]

        sym_z_shape = (z_numrows, cols) if z_subset is not None else (rows, cols)
        sym_xmat_shape = (rows, cols)
        sym_dmask_shape = x0.shape
        sym_mu_shape = (rows,)
        sym_rsigma_shape = (rows,)
        sym_shapes = [
            sym_z_shape,
            sym_xmat_shape,
            sym_dmask_shape,
            sym_mu_shape,
            sym_rsigma_shape,
        ]

        return map_to_symbolic_tensors(outputs, sym_shapes)


@register_symbolic_op(dropout_layer_norm, "dropout_add_ln_bwd")
class SymbolicDropoutAddLnBwd(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        dz: torch.Tensor,
        dx: Optional[torch.Tensor],
        x: torch.Tensor,
        x0: Optional[torch.Tensor],
        dmask: Optional[torch.Tensor],
        mu: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        rowscale: Optional[torch.Tensor],
        colscale: Optional[torch.Tensor],
        x0_subset: Optional[torch.Tensor],
        z_subset: Optional[torch.Tensor],
        dropout_p: float,
        rowscale_const: float,
        x0_numrows: int,
        has_residual: bool,
        is_rms_norm: bool = False,
    ):
        # outputs: dx0, dresidual, dgamma, dbeta, dgamma_part, dbeta_part, Optional(dcolscale), Optional(drowscale)
        # ----------------
        # dz:           (B * S, H)
        # dx:           (B * S, H) | None
        # x:            (B * S, H)
        # x0:           (B * S, H) | None
        # dmask:        (B * S, H) | None
        # mu:           (B * S,)
        # rsigma:       (B * S,)
        # gamma:        (H,)
        # rowscale:     (B * S,)   | None
        # colscale:     (H,)       | None
        # x0_subset:    (B * S)    | None
        # z_subset:     (B * S)    | None
        assert len(outputs) == 6 or len(outputs) == 8

        rows = x0_numrows if x0_subset is not None else x.shape[0]
        cols = x.shape[1]

        sym_dx0_shape = (rows, cols)
        sym_dresidual_shape = x.shape if has_residual else None
        sym_dgamma_shape = gamma.shape
        sym_dbeta_shape = gamma.shape
        sym_dgamma_part_shape = (outputs[4].shape[0], cols)
        sym_dbeta_part_shape = (outputs[5].shape[0], cols)

        sym_shapes = [
            sym_dx0_shape,
            sym_dresidual_shape,
            sym_dgamma_shape,
            sym_dbeta_shape,
            sym_dgamma_part_shape,
            sym_dbeta_part_shape,
        ]
        if len(outputs) == 8:
            sym_dcolscale_shape = (cols,)
            sym_dcolscale_part_shape = (outputs[6].shape[0], cols)
            sym_shapes += [
                sym_dcolscale_shape,
                sym_dcolscale_part_shape,
            ]

        return map_to_symbolic_tensors(outputs, sym_shapes)


@register_symbolic_op(dropout_layer_norm, "dropout_add_ln_parallel_residual_fwd")
class SymbolicDropoutAddLnParallelResidualFwd(SymbolicOp):
    @staticmethod
    def apply(
        outputs,
        x0: torch.Tensor,
        x1: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        gamma0: torch.Tensor,
        beta0: Optional[torch.Tensor],
        gamma1: Optional[torch.Tensor],
        beta1: Optional[torch.Tensor],
        dropout_p: float,
        eps: float,
        gen_,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
    ):
        # outputs: zmat, xmat, dmask, mu, rsigma
        # ----------------
        # x0:           (B * S, H)
        # x1:           (B * S, H) | None
        # residual:     (B * S, H) | None
        # gamma0:       (H,)
        # beta1:        (H,)       | None
        # gamma1:       (H,)       | None
        # beta1:        (H,)       | None

        assert x0.ndim == 2
        rows, cols = x0.shape

        sym_z_shape = (rows, cols)
        sym_xmat_shape = (rows, cols)
        sym_dmask_shape = x0.shape
        sym_mu_shape = (rows,)
        sym_rsigma_shape = (rows,)
        sym_shapes = [
            sym_z_shape,
            sym_z_shape,
            sym_xmat_shape,
            sym_dmask_shape,
            sym_dmask_shape,
            sym_mu_shape,
            sym_rsigma_shape,
        ]

        return map_to_symbolic_tensors(outputs, sym_shapes)


@register_symbolic_op(activations, "swiglu_fwd")
class SymbolicSwiGLUFwd(SymbolicOp):
    @staticmethod
    def apply(outputs, x: torch.Tensor, y: torch.Tensor):
        assert x.shape == y.shape
        sym_output_shape = x.shape
        return SymbolicTensor(outputs, sym_output_shape)


@register_symbolic_op(activations, "swiglu_bwd")
class SymbolicSwiGLUBwd(SymbolicOp):
    @staticmethod
    def apply(outputs, x: torch.Tensor, y: torch.Tensor, g: torch.Tensor):
        assert x.shape == y.shape
        sym_shapes = [
            x.shape,
            y.shape,
        ]
        return map_to_symbolic_tensors(outputs, sym_shapes)


# @register_symbolic_op(losses, "vocab_parallel_cross_entropy")
# class SymbolicVocabParallelCrossEntropy(SymbolicOp):
#     @staticmethod
#     def apply(
#         outputs,
#         input: torch.Tensor,
#         target: torch.Tensor,
#         process_group,
#         label_smoothing: float = 0.0,
#     ):
#         assert input.ndim == 2, f"input should be 2D, got {input.ndim}"
#         ret = SymbolicTensor(outputs, symbolic_shape=(input.shape[0],))
#         if input.requires_grad:
#             _exp_logits = torch.empty_like(input)
#             _target_mask = torch.empty_like(input, dtype=torch.bool)
#             _masked_target_1d = torch.empty_like(input).view(-1)
#             ret.context = SymbolicOpContext(
#                 op=SymbolicVocabParallelCrossEntropy,
#                 saved_tensors=(_exp_logits, _target_mask, _masked_target_1d),
#             )
#         return ret


@register_symbolic_op(mist.utils.memory, "_kernel_make_viewless_tensor")
class SymbolicMakeViewlessTensor(SymbolicOp):
    @staticmethod
    def apply(outputs, input: torch.Tensor, requires_grad: bool):
        return SymbolicTensor(outputs, symbolic_shape=input.shape)


if scaled_upper_triang_masked_softmax_cuda is not None:

    @register_symbolic_op(scaled_upper_triang_masked_softmax_cuda, "forward")
    class SymbolicScaledUpperTriangMaskedSoftmaxFwd(SymbolicOp):
        @staticmethod
        def apply(
            outputs,
            input: torch.Tensor,
            *args,
            **kwargs,
        ):
            assert input.ndim == 3, f"input should be 3D, got {input.ndim}"
            return SymbolicTensor(outputs, symbolic_shape=input.shape)

    @register_symbolic_op(scaled_upper_triang_masked_softmax_cuda, "backward")
    class SymbolicScaledUpperTriangMaskedSoftmaxBwd(SymbolicOp):
        @staticmethod
        def apply(
            outputs,
            grad: torch.Tensor,
            *args,
            **kwargs,
        ):
            assert grad.ndim == 3, f"grad should be 3D, got {grad.ndim}"
            outputs = make_viewless_tensor(
                outputs,
                requires_grad=outputs.requires_grad,
                keep_graph=outputs.requires_grad,
            )
            return SymbolicTensor(outputs, symbolic_shape=grad.shape)


if scaled_masked_softmax_cuda is not None:

    @register_symbolic_op(scaled_masked_softmax_cuda, "forward")
    class SymbolicScaledMaskedSoftmaxFwd(SymbolicOp):
        @staticmethod
        def apply(
            outputs,
            input: torch.Tensor,
            *args,
            **kwargs,
        ):
            assert input.ndim == 3, f"input should be 3D, got {input.ndim}"
            return SymbolicTensor(outputs, symbolic_shape=input.shape)

    @register_symbolic_op(scaled_masked_softmax_cuda, "backward")
    class SymbolicScaledMaskedSoftmaxBwd(SymbolicOp):
        @staticmethod
        def apply(
            outputs,
            grad: torch.Tensor,
            *args,
            **kwargs,
        ):
            assert grad.ndim == 3, f"grad should be 3D, got {grad.ndim}"
            outputs = make_viewless_tensor(
                outputs,
                requires_grad=outputs.requires_grad,
                keep_graph=outputs.requires_grad,
            )
            return SymbolicTensor(outputs, symbolic_shape=grad.shape)
