import sys
import functools

import torch

import mist
from mist.overrides import override_item, register_overriden_func, override_attr
from mist.sym_torch.registry import map_symbols_to_concretes, _deal_with_proxy
from mist.utils.tensor_entry import tensor_to_entry

ori_autogrd_func_apply = torch.autograd.Function.apply.__func__


def symbolic_compatible(func):
    if type(func).__name__ == "staticmethod":
        func.__func__.is_symbolic_compatible = True
    else:
        func.is_symbolic_compatible = True
    return func


# @register_overriden_func(torch.autograd.Function, "apply")
# @classmethod
# def autograd_func_apply_with_sym_check(cls, *args, **kwargs):
#     # Deal with proxy (which is not needed if you don't use tracing)
#     proxy_found, ret = _deal_with_proxy(wrapped_function, *args, **kwargs)
#     if proxy_found:
#         return ret

#     found, concrete_args, concrete_kwargs = map_symbols_to_concretes(args, kwargs)
#     if found:
#         if getattr(cls.forward, "is_symbolic_compatible", False):
#             ret = ori_autogrd_func_apply(cls, *args, **kwargs)
#             return ret
#         else:
#             try:
#                 ret = ori_autogrd_func_apply(cls, *args, **kwargs)
#                 ret_info = (
#                     ret if not isinstance(ret, torch.Tensor) else tensor_to_entry(ret)
#                 )
#             except Exception as e:
#                 ret_info = "Exception: " + str(e)
#             raise RuntimeError(
#                 f"Inputs to autograd.Function [{cls}] contains symbols. "
#                 f"This could happen when this op is not overriden. "
#                 f"See the stack trace above to find the corresponding op.\n"
#                 f"    - Func: {cls}\n"
#                 f"    - Args: {args}\n"
#                 f"    - Kwargs: {kwargs}\n"
#                 f"    - Output/Exception: {ret_info}"
#             )
#     else:
#         return ori_autogrd_func_apply(cls, *args, **kwargs)


# TODO(zhanda): remove this after all tests are passed
# Below are for debugging
class F(torch.autograd.Function):
    @staticmethod
    def forward(ctx):
        print("2")
        pass


if __name__ == "__main__":
    F.apply()

    from mist.overrides import MistRevertPatcher

    with MistRevertPatcher():
        print("aaa")

    print("111")
