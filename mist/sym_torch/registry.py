from typing import Dict, Tuple, Any, Set, List, Optional, Union, Callable
from functools import wraps
from importlib import import_module
from numbers import Number

import sympy as sp
import torch
import torch.fx as fx
from torch.utils._pytree import tree_map

from mist import global_symbol_manager as gsm
from mist.overrides import register_overriden_func, get_ori_torch_op
from mist.sym_torch.symbolic_tensor import SymbolicTensor
from mist.utils.tracing_patcher import get_patchers


def move_to_device(pytree, device):
    def fn(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    return tree_map(fn, pytree)


def _has_obj_of_class(cls, args, kwargs):
    found = False

    def fn(x):
        nonlocal found
        if isinstance(x, cls):
            found = True

    tree_map(fn, args)
    tree_map(fn, kwargs)
    return found


def _deal_with_proxy(func, *args, **kwargs):
    """
    This is the case to deal with the proxy function.
    Or in the future, it can be used to deal with other __torch_function__ overrides.

    The logic is:
    1. If there is proxy in args or kwargs, then we need to first patch to the proxy's __torch_function__.
    2. When being patched to proxy's __torch_function__, the op will be called again with the proxies
       replaced by the symbolic/torch tensors.
    3. Then we can use the original logic for symbolic tensor.
    """
    found = _has_obj_of_class(fx.proxy.Proxy, args, kwargs)
    ret = None
    if found:
        ret = fx.proxy.Proxy.__torch_function__(func, None, args, kwargs)
    if found and len(get_patchers()) == 0:
        raise RuntimeError(
            "The proxy is only used for tracing, "
            f"but there is no patcher registered.\n"
            f"    - Func: {func}\n"
            f"    - Args: {args}\n"
            f"    - Kwargs: {kwargs}\n"
            f"    - Ret: {ret}\n"
        )
    return found, ret


def map_symbols_to_concretes_single(obj):
    found = False

    def fn(x):
        nonlocal found
        if isinstance(x, SymbolicTensor):
            if not x.check_complete():
                raise RuntimeError(
                    f"Symbolic tensor {x} is not complete. Obj: {obj}. "
                    "This is mainly due to the parent op is not overriden. "
                    "The parent op get a symbolic tensor as input and "
                    "return a symbolic tensor without symbolic shape assigned."
                )
            found = True
            return x.to_torch_tensor()
        if isinstance(x, sp.Basic):
            if len(x.free_symbols) > 0:
                assert all(
                    isinstance(s, sp.Symbol) for s in x.free_symbols
                ), f"Only support single symbol in expression, got {x}"
            found = True
            ret = gsm.subs(x)
            if not isinstance(ret, Number):
                try:
                    ret = float(ret)
                except TypeError:
                    raise ValueError(
                        f"{x} is not a number after substitution, perhaps the concrete value is not assigned."
                    )
            return ret
        return x

    concrete_obj = tree_map(fn, obj)

    return found, concrete_obj


def map_symbols_to_concretes(args, kwargs):
    found = False

    def fn(x):
        nonlocal found
        if isinstance(x, SymbolicTensor):
            if not x.check_complete():
                raise RuntimeError(
                    f"Symbolic tensor {x} is not complete. Args: {args}, kwargs: {kwargs}. "
                    "This is mainly due to the parent op is not overriden. "
                    "The parent op get a symbolic tensor as input and "
                    "return a symbolic tensor without symbolic shape assigned."
                )
            found = True
            return x.to_torch_tensor()
        if isinstance(x, sp.Basic):
            if len(x.free_symbols) > 0:
                assert all(
                    isinstance(s, sp.Symbol) for s in x.free_symbols
                ), f"Only support single symbol in expression, got {x}"
            found = True
            ret = gsm.subs(x)
            if not isinstance(ret, Number):
                try:
                    ret = float(ret)
                except TypeError:
                    raise ValueError(
                        f"{x} is not a number after substitution, perhaps the concrete value is not assigned."
                    )
            return ret
        return x

    args = tree_map(fn, args)
    kwargs = tree_map(fn, kwargs)

    return found, args, kwargs


def register_symbolic_op(
    parent: Callable,
    name: str,
):
    def decorator(cls):
        torch_op = get_ori_torch_op(parent, name)

        def wrapped_function(*args, **kwargs):
            # Deal with proxy (which is not needed if you don't use tracing)
            proxy_found, ret = _deal_with_proxy(wrapped_function, *args, **kwargs)
            if proxy_found:
                return ret

            # Map symbols to concretes, and map symbolic tensors to torch tensors
            symbol_found, concrete_args, concrete_kwargs = map_symbols_to_concretes(
                args, kwargs
            )

            if not symbol_found:
                return torch_op(*args, **kwargs)

            else:
                # Call preprocess e.g. move_to_with_grad_tracked
                preprocess = getattr(cls, "preprocess", None)
                if preprocess:
                    concrete_args, concrete_kwargs = preprocess(
                        concrete_args, concrete_kwargs
                    )

                # Call the original torch op
                outputs = torch_op(*concrete_args, **concrete_kwargs)

                # Call postprocess e.g. move_to_with_grad_tracked
                postprocess = getattr(cls, "postprocess", None)
                if postprocess:
                    outputs = postprocess(outputs)

                # Assign symbolic shape to the outputs
                return cls.apply(outputs, *args, **kwargs)

        wrapped_function.is_symbolic_compatible = True

        register_overriden_func(parent=parent, name=name)(wrapped_function)

        return cls

    return decorator
