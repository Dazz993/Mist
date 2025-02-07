import einops
import torch
from torch.fx.proxy import Proxy

from mist.overrides.base import register_overriden_func, override_attr


@register_overriden_func(einops._backends.TorchBackend, "is_appropriate_type")
def einops_backend_is_appropriate_type(self, tensor):
    return isinstance(tensor, torch.Tensor) or isinstance(tensor, Proxy)
