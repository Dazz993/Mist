from apex.transformer.functional import fused_softmax as apex_fused_softmax
from apex.transformer.enums import AttnMaskType
from apex._autocast_utils import _cast_if_autocast_enabled
import torch
from mist.overrides import register_overriden_func
from mist.utils.memory import make_viewless_tensor


def fused_scale_mask_softmax(
    input,
    mask,
    input_inf_fp16=True,
    input_inf_bf16=False,
    attn_mask_type=AttnMaskType.causal,
    scaled_masked_softmax_fusion=True,
    mask_func=None,
    softmax_in_fp32=False,
    scale=None,
):
    if scale is not None:
        softmax_in_fp32 = True

    softmax = apex_fused_softmax.FusedScaleMaskSoftmax(
        input_in_fp16=input_inf_fp16,
        input_in_bf16=input_inf_bf16,
        attn_mask_type=attn_mask_type,
        scaled_masked_softmax_fusion=scaled_masked_softmax_fusion,
        mask_func=mask_func,
        softmax_in_fp32=softmax_in_fp32,
        scale=scale,
    )

    return softmax(input, mask)


def scaled_upper_triang_masked_softmax_fn(inputs, scale):
    return apex_fused_softmax.ScaledUpperTriangMaskedSoftmax.apply(inputs, scale)


def scaled_masked_softmax_fn(inputs, mask, scale):
    return apex_fused_softmax.ScaledMaskedSoftmax.apply(inputs, mask, scale)


def scaled_softmax_fn(inputs, scale):
    return apex_fused_softmax.ScaledSoftmax.apply(inputs, scale)


@register_overriden_func(apex_fused_softmax, "scaled_upper_triang_masked_softmax")
def scaled_upper_triang_masked_softmax(inputs, _, scale):
    b, np, sq, sk = inputs.size()
    assert sq == sk, "causal mask is only for self attention"
    # Reshaping input to 3D tensor (attn_batches, sq, sk)
    inputs = inputs.view(-1, sq, sk)
    args = _cast_if_autocast_enabled(inputs, scale)
    with torch.cuda.amp.autocast(enabled=False):
        probs = scaled_upper_triang_masked_softmax_fn(*args)
    return probs.view(b, np, sq, sk)


@register_overriden_func(apex_fused_softmax, "scaled_masked_softmax")
def scaled_masked_softmax(inputs, mask, scale):
    # input is 4D tensor (b, np, sq, sk)
    if mask is not None:
        args = _cast_if_autocast_enabled(inputs, mask, scale)
        with torch.cuda.amp.autocast(enabled=False):
            return scaled_masked_softmax_fn(*args)
    else:
        args = _cast_if_autocast_enabled(inputs, scale)
        with torch.cuda.amp.autocast(enabled=False):
            return scaled_softmax_fn(*args)


torch.fx.wrap("scaled_upper_triang_masked_softmax_fn")
torch.fx.wrap("scaled_masked_softmax_fn")
torch.fx.wrap("scaled_softmax_fn")
