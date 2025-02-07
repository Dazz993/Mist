from __future__ import annotations
from numbers import Number
from typing import Any, Union, Sequence, Tuple, Optional, Dict, List
import contextlib

import sympy as sp
from sympy.core.decorators import (
    call_highest_priority,
    sympify_method_args,
    sympify_return,
)
from torch.utils._pytree import tree_map

from mist.overrides import override_attr
from mist.utils.fx import register_slice_in_pytree_map

register_slice_in_pytree_map()

global_symbol_manager = None

SymbolicType = Union[sp.Basic, Number]


def _subs_fn(expr, mapping: Dict[str, Any], return_number_if_possible: bool = True):
    """
    Substitute the symbols in the expression with the values in the mapping,
    and return the number if possible based on the flag.
    """
    if isinstance(expr, sp.Basic):
        ret = expr.subs(mapping)
        if return_number_if_possible:
            if isinstance(ret, sp.Integer):
                return int(ret)
            elif isinstance(ret, sp.Float):
                return float(ret)
        return ret

    return expr


DEFAULT_SP_KWARGS = {
    "real": True,
    "nonnegative": True,
}


class SymbolManager:
    def __init__(self, **sp_kwargs) -> None:
        self.sp_kwargs = {**DEFAULT_SP_KWARGS, **sp_kwargs}
        self.mapping: Dict[sp.Symbol, Number] = {}
        self.name2symbol: Dict[str, sp.Symbol] = {}
        self._memo_for_fast_subs: Dict[sp.Basic, Number] = {}

        # Setup naive symbols
        self.x = self.symbols("dummy_x", 4, integer=True, positive=True)
        self.y = self.symbols("dummy_y", 128, integer=True, positive=True)
        self.z = self.symbols("dummy_z", 768, integer=True, positive=True)
        self.xyz = (self.x, self.y, self.z)

    def reset(self):
        self.mapping: Dict[sp.Symbol, Number] = {}
        self._memo_for_fast_subs: Dict[sp.Basic, Number] = {}

    def _preprocess_exprs_and_concretes(
        self,
        exprs: Union[SymbolicType, Sequence[SymbolicType]],
        concretes: Union[Number, Sequence[Number]],
    ):
        if isinstance(exprs, (sp.Basic, Number)) and isinstance(
            concretes, (sp.Basic, Number)
        ):
            exprs = [exprs]
            concretes = [concretes]

        if not isinstance(exprs, Sequence) or not isinstance(concretes, Sequence):
            raise ValueError(f"Get wrong input type: {type(exprs)} {type(concretes)}")

        if len(exprs) != len(concretes):
            raise ValueError(
                f"Diff length: {len(exprs)} {len(concretes)}. exprs={exprs}, concretes={concretes}"
            )

        return exprs, concretes

    def symbols(
        self,
        names: Union[str, Sequence[str]],
        concretes: Optional[Union[Number, Sequence[Number]]] = None,
        **kwargs,
    ) -> Union[sp.Symbol, Sequence[sp.Symbol]]:
        """
        Create symbols with the given names and map them to the given concretes if provided.

        Parameters
        ----------
        names : Union[str, Sequence[str]]
            names of the symbols (used in SymPy)
        concretes : Optional[Union[Number, Sequence[Number]]], optional
            concrete values of the symbols, by default None

        Returns
        -------
        Union[sp.Symbol, Sequence[sp.Symbol]]
            the created symbols
        """
        sp_kwargs = {**self.sp_kwargs, **kwargs}
        ret = sp.symbols(names, **sp_kwargs)
        if concretes is not None:
            self.map(ret, concretes)
        return ret

    def map(
        self,
        exprs: Union[SymbolicType, Sequence[SymbolicType]],
        concretes: Union[Number, Sequence[Number]],
    ):
        """
        Map the given expressions to the given concretes.
        Add the mapping to `self.mapping`.

        Parameters
        ----------
        exprs : Union[SymbolicType, Sequence[SymbolicType]]
            expressions to map
        concretes : Union[Number, Sequence[Number]]
            concrete values of the expressions
        """
        exprs, concretes = self._preprocess_exprs_and_concretes(exprs, concretes)

        for expr, concrete in zip(exprs, concretes):
            if isinstance(expr, Number):
                assert (
                    expr == concrete
                ), f"expr {expr} != concrete {concrete}. exprs: {exprs}, concretes: {concretes}"
                continue

            # Fast path
            if isinstance(expr, sp.Symbol):
                if expr in self.mapping:
                    assert self.mapping[expr] == concrete, (
                        f"expr {expr} has been mapped to {self.mapping[expr]}, "
                        f"but now it's mapped to {concretes}"
                    )
                else:
                    self.mapping[expr] = concrete
                    self.name2symbol[expr.name] = expr
                continue

            expr = self.subs(expr)
            len_free_symbols = (
                len(expr.free_symbols) if isinstance(expr, sp.Basic) else 0
            )
            if len_free_symbols == 0:
                self.verify(expr, concrete)
            elif len_free_symbols == 1:
                free_symbol, solution = self.solve(expr, concrete)
                self.mapping[free_symbol] = solution
                self.name2symbol[free_symbol.name] = free_symbol
            else:
                raise ValueError(f"More than one free symbol: {expr}")

    def verify(
        self,
        exprs: Union[SymbolicType, Sequence[SymbolicType]],
        concretes: Union[Number, Sequence[Number]],
    ):
        """
        Verify the given expressions match the given concretes.

        Parameters
        ----------
        exprs : Union[SymbolicType, Sequence[SymbolicType]]
            expressions to verify
        concretes : Union[Number, Sequence[Number]]
            concrete values of the expressions
        """
        exprs, concretes = self._preprocess_exprs_and_concretes(exprs, concretes)

        for expr, concrete in zip(exprs, concretes):
            concrete_expr = self.subs(expr)
            assert concrete_expr == concrete, (
                f"Expr {expr} with concrete {concrete} "
                f"has concrete expr {concrete_expr}"
            )

    def solve(self, expr: SymbolicType, concrete: Number):
        """
        Solve the given expression with the given concrete value.
        Only work if the expression has only one free symbol.

        Parameters
        ----------
        expr : SymbolicType
            expression to solve
        concrete : Number
            concrete value of the expression

        Returns
        -------
        Tuple[sp.Symbol, Number]
            the free symbol and the solution
        """
        equation = sp.Eq(expr, concrete)
        equation_with_values = equation.subs(self.mapping)
        assert len(equation_with_values.free_symbols) == 1, (
            f"Equation '{equation}' with values '{equation_with_values}' "
            f"has more than one free symbol"
        )
        free_symbol = list(equation_with_values.free_symbols)[0]
        solution = sp.solve(equation_with_values)
        assert len(solution) == 1, f"More than one solution: {solution}"
        return free_symbol, solution[0]

    def subs(self, expr, mapping: Optional[Dict[sp.Symbol, Number]] = None):
        """
        Substitute the symbols in the expression with the values in the mapping.
        It's an extension of `sympy.subs` that provides more functionalities.
        """
        mapping = mapping or self.mapping
        return tree_map(
            lambda x: _subs_fn(x, mapping, return_number_if_possible=True), expr
        )

    def load_from_symbol_manager(self, other: SymbolManager):
        for name in self.__dict__:
            self.__dict__[name] = other.__dict__[name]

    def __contains__(self, item: SymbolicType):
        if isinstance(item, Number):
            return True
        elif not isinstance(item, sp.Basic):
            raise ValueError(f"Get wrong input type: {type(item)}")
        return all(symbol in self.mapping for symbol in item.free_symbols)


global_symbol_manager = gsm = SymbolManager()

# ===========================================================================
# Override logics for sympy
# ===========================================================================

_ori_sp_symbol_slots = sp.Symbol.__slots__
_ori_sp_basic_eq = sp.core.basic.Basic.__eq__
_ori_sp_basic_ne = sp.core.basic.Basic.__ne__
_ori_relational_bool = sp.core.relational.Relational.__bool__
_ori_core_expr_int = sp.core.expr.Expr.__int__
_ori_core_expr_float = sp.core.expr.Expr.__float__
_ori_core_expr_floordiv = sp.core.expr.Expr.__floordiv__
_ori_core_expr_rfloordiv = sp.core.expr.Expr.__rfloordiv__


def override_eq_ne():
    sp.core.basic.Basic.__eq__ = _basic_eq
    sp.core.basic.Basic.__ne__ = _basic_ne


def reset_eq_ne():
    sp.core.basic.Basic.__eq__ = _ori_sp_basic_eq
    sp.core.basic.Basic.__ne__ = _ori_sp_basic_ne


# Define the context manager for overriding the default `sympy` behaviors
@contextlib.contextmanager
def temporarily_set_sp_eq_ne():
    override_eq_ne()
    yield
    reset_eq_ne()


def _basic_eq(self, other):
    """
    Override the default `sympy.basic.Basic.__eq__` to make it
    return a concrete value if possible.
    """

    if isinstance(other, Number):
        if self in gsm._memo_for_fast_subs:
            return gsm._memo_for_fast_subs[self] == other
        else:
            reset_eq_ne()
            value = gsm.subs(self)
            override_eq_ne()
            if isinstance(value, Number):
                gsm._memo_for_fast_subs[self] = value
                return value == other
    return _ori_sp_basic_eq(self, other)


def _basic_ne(self, other):
    """
    Override the default `sympy.basic.Basic.__ne__` to make it
    return a concrete value if possible.
    """

    if isinstance(other, Number):
        if self in gsm._memo_for_fast_subs:
            return gsm._memo_for_fast_subs[self] != other
        else:
            reset_eq_ne()
            value = gsm.subs(self)
            override_eq_ne()
            if isinstance(value, Number):
                gsm._memo_for_fast_subs[self] = value
                return value != other
    return _ori_sp_basic_ne(self, other)


def _relational_bool(self):
    """
    Override the default `sympy.relational.Relational.__bool__` to make it
    return a concrete value if possible.
    """
    if self.is_Relational:
        if not all(s in global_symbol_manager for s in self.free_symbols):
            raise TypeError("cannot determine truth value of Relational")
        return bool(self.subs(global_symbol_manager.mapping))
    return bool(self)


def _expr_int(self):
    """
    Override the default `sympy.core.expr.Expr.__int__` to make it
    return a concrete value if possible.
    """
    if isinstance(self, Number):
        return int(self)
    subbed = gsm.subs(self)
    if not isinstance(subbed, Number):
        raise ValueError(f"Cannot convert {self} to int")
    return int(subbed)


def _expr_float(self):
    """
    Override the default `sympy.core.expr.Expr.__float__` to make it
    return a concrete value if possible.
    """
    if isinstance(self.evalf(), Number):
        return float(self.evalf())
    subbed = gsm.subs(self)
    if isinstance(subbed, Number):
        return float(subbed)
    subbed = self.evalf()
    if not isinstance(subbed, Number):
        raise ValueError(f"Cannot convert {self} to float")
    return float(subbed)


# @call_highest_priority("__rfloordiv__")
# def _expr_floordiv(self, other):
#     out = _ori_core_expr_floordiv(self, other)
#     if isinstance(out, sp.Basic) and out.has(sp.functions.elementary.integers.floor):
#         return self / other
#     else:
#         return out


# @call_highest_priority("__floordiv__")
# def _expr_rfloordiv(self, other):
#     return _expr_floordiv(other, self)


sp.Symbol.__slots__ = _ori_sp_symbol_slots + ("_concrete_value",)
override_attr(sp.core.relational.Relational, "__bool__", _relational_bool)
# override_attr(sp.core.basic.Basic, "__eq__", _basic_eq)
# override_attr(sp.core.basic.Basic, "__ne__", _basic_ne)
override_attr(sp.core.expr.Expr, "__int__", _expr_int)
override_attr(sp.core.expr.Expr, "__float__", _expr_float)
# override_attr(sp.core.expr.Expr, "__floordiv__", _expr_floordiv)
# override_attr(sp.core.expr.Expr, "__rfloordiv__", _expr_rfloordiv)

if __name__ == "__main__":
    from mist import gsm

    x, y, z = gsm.xyz

    qq = x // 4
    ww = x // y
