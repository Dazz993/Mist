import torch
import torch.nn.functional as F
from functools import partial

from mist.config import ModelConfig
from mist.modules import LayerNorm, RMSNorm
from mist.modules.activations import sqrelu_fwd


def get_norm(
    model_config: ModelConfig,
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    normalization = model_config.normalization
    hidden_size = model_config.hidden_size
    eps = model_config.normalization_eps
    sequence_parallel = model_config.sequence_parallel
    if normalization == "layernorm":
        return LayerNorm(
            hidden_size,
            eps=eps,
            # sequence_parallel=sequence_parallel,
            device=device,
            dtype=dtype,
        )
    elif normalization == "rmsnorm":
        return RMSNorm(
            hidden_size,
            eps=eps,
            # sequence_parallel=sequence_parallel,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Normalization {normalization} not supported.")


def get_activation(activation: str):
    if activation in ("gelu", "geglu"):
        return F.gelu
    elif activation in ("relu",):
        return partial(F.relu, inplace=True)
    elif activation in ("sigmoid", "glu"):
        return F.sigmoid
    elif activation in ("gelu_fast", "gelu_approx", "gelu_pytorch_tanh"):
        return partial(F.gelu, approximate="tanh")
    elif activation == "sqrelu":
        return sqrelu_fwd
    elif activation in ("silu", "swiglu"):
        return F.silu
    else:
        raise ValueError(f"Activation {activation} not supported.")
