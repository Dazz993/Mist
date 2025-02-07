import torch

from mist.overrides.base import (
    register_overriden_func,
    MistRevertPatcher,
)


@register_overriden_func(torch.jit, "script")
def torch_jit_script(*args, **kwargs):
    with MistRevertPatcher():
        ret = torch.jit.script(*args, **kwargs)
    return ret