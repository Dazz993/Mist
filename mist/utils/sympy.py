"""
This module contains helper functions for sympy.

1. math functions to support both sympy and python numbers
"""

import math
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Iterator
from functools import cache
from numba import njit, jit, prange

import sympy as sp

# Some version of sympy may not work. We currently use 1.10.0
try:
    from sympy.utilities import autowrap
except ImportError:
    autowrap = None


def indicator(condition):
    if condition is True:
        return 1
    return sp.Piecewise((1, condition), (0, True))


def fake_floordiv(a: Union[int, sp.Basic], b: Union[int, sp.Basic]):
    """
    floor division for sympy. It is mainly used to replace the default floor
    division with the division operator. The reason is to avoid the floor in
    the output expression.

    Parameters
    ----------
    a
        a // b as a
    b
        a // b as b

    Returns
    -------
        the floor division of a and b (with the floor removed if possible)
    """
    out = a // b

    if isinstance(out, sp.Basic) and out.has(sp.functions.elementary.integers.floor):
        return a / b
    else:
        return floordiv(a, b)


def floordiv(a: Union[Number, sp.Basic], b: Union[Number, sp.Basic]):
    out = a // b

    if isinstance(out, Number):
        return sp2py(out)
    else:
        return out


def floor(a: Union[Number, sp.Basic]):
    out = sp.functions.elementary.integers.floor(a)

    if isinstance(out, Number):
        return sp2py(out)
    else:
        return out


def ceil(a: Union[Number, sp.Basic]):
    out = sp.functions.elementary.integers.ceiling(a)

    if isinstance(out, Number):
        return sp2py(out)
    else:
        return out


def sqrt(a: Union[Number, sp.Basic]):
    out = sp.functions.elementary.complexes.sqrt(a)

    try:
        out = float(out)
    except TypeError:
        pass

    if isinstance(out, Number):
        return sp2py(out)
    else:
        return out


def sp2py(obj):
    if isinstance(obj, sp.core.numbers.Float):
        return float(obj)
    elif isinstance(obj, sp.core.numbers.Integer):
        return int(obj)
    else:
        return obj


ORI_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
}


def replace_args_with_float_args(args):
    changed = False
    float_args = []
    for arg in args:
        if not arg.is_integer:
            float_args.append(arg)
        else:
            float_args.append(sp.Symbol(f"_float_{str(arg)}", real=True))
            changed = True
    return changed, tuple(float_args)


@cache
def autowrap_with_cython(args, expr, **kwargs):
    changed, float_args = replace_args_with_float_args(args)
    if changed:
        expr = expr.subs({arg: float_arg for arg, float_arg in zip(args, float_args)})
    return autowrap.autowrap(
        expr, args=float_args, language="C", backend="cython", **kwargs
    )


@cache
def ufuncify_with_cython(args, expr, **kwargs):
    changed, float_args = replace_args_with_float_args(args)
    if changed:
        expr = expr.subs({arg: float_arg for arg, float_arg in zip(args, float_args)})
    # return autowrap.ufuncify(float_args, expr, language="C", backend="numpy", **kwargs)
    # return autowrap.ufuncify(float_args, expr, language="C", backend="cython", **kwargs)
    return jit(sp.lambdify(args=float_args, expr=expr, modules="numpy", **kwargs))


@cache
def lambdify_with_numpy(args, expr, **kwargs):
    changed, float_args = replace_args_with_float_args(args)
    if changed:
        expr = expr.subs({arg: float_arg for arg, float_arg in zip(args, float_args)})
    return sp.lambdify(args=float_args, expr=expr, modules="numpy", **kwargs)
