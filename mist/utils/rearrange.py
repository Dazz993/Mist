import functools
from collections import OrderedDict
from typing import Sequence, Union, Tuple, Dict, List, Optional, Any

import numpy as np
import sympy as sp
import einops
from einops.parsing import ParsedExpression, AnonymousAxis, _ellipsis
from einops.einops import (
    EinopsError,
    TransformRecipe,
    CookedRecipe,
    is_ellipsis_not_in_parenthesis,
    _product,
)

from mist import gsm
from mist.utils.sympy import fake_floordiv

_unknown_axis_length = -999999


def prod(a):
    if len(a) == 0:
        return 1
    else:
        return np.prod(a)


def _reconstruct_from_shape_uncached(
    self: TransformRecipe, shape: List[int]
) -> CookedRecipe:
    """
    Reconstruct all actual parameters using shape.
    Shape is a tuple that may contain integers, shape symbols (tf, keras, theano) and UnknownSize (keras, mxnet)
    known axes can be integers or symbols, but not Nones.
    """
    axes_lengths: List[int] = list(self.elementary_axes_lengths)
    if self.ellipsis_position_in_lhs != 10000:
        if len(shape) < len(self.input_composite_axes) - 1:
            raise EinopsError(
                "Expected at least {} dimensions, got {}".format(
                    len(self.input_composite_axes) - 1, len(shape)
                )
            )
    else:
        if len(shape) != len(self.input_composite_axes):
            raise EinopsError(
                "Expected {} dimensions, got {}".format(
                    len(self.input_composite_axes), len(shape)
                )
            )

    ellipsis_shape: List[int] = []
    for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composite_axes):
        before_ellipsis = input_axis
        after_ellipsis = input_axis + len(shape) - len(self.input_composite_axes)
        if input_axis == self.ellipsis_position_in_lhs:
            assert len(known_axes) == 0 and len(unknown_axes) == 1
            unknown_axis: int = unknown_axes[0]
            ellipsis_shape = shape[before_ellipsis : after_ellipsis + 1]
            for d in ellipsis_shape:
                if d is None:
                    raise EinopsError(
                        "Couldn't infer shape for one or more axes represented by ellipsis"
                    )
            total_dim_size: int = _product(ellipsis_shape)
            axes_lengths[unknown_axis] = total_dim_size
        else:
            if input_axis < self.ellipsis_position_in_lhs:
                length = shape[before_ellipsis]
            else:
                length = shape[after_ellipsis]
            known_product = 1
            for axis in known_axes:
                known_product *= axes_lengths[axis]

            if len(unknown_axes) == 0:
                if (
                    isinstance(length, int)
                    and isinstance(known_product, int)
                    and length != known_product
                ):
                    raise EinopsError(
                        "Shape mismatch, {} != {}".format(length, known_product)
                    )
            # this is enforced when recipe is created
            # elif len(unknown_axes) > 1:
            #     raise EinopsError(
            #         "Lengths of two or more axes in parenthesis not provided (dim={}), can't infer dimensions".
            #             format(known_product)
            #     )
            else:
                if (
                    isinstance(length, int)
                    and isinstance(known_product, int)
                    and length % known_product != 0
                ):
                    raise EinopsError(
                        "Shape mismatch, can't divide axis of length {} in chunks of {}".format(
                            length, known_product
                        )
                    )

                unknown_axis = unknown_axes[0]
                # [Zhanda] Changed to fake_floordiv
                # inferred_length: int = length // known_product
                inferred_length = fake_floordiv(length, known_product)
                axes_lengths[unknown_axis] = inferred_length

    # at this point all axes_lengths are computed (either have values or variables, but not Nones)

    # TODO more readable expression
    init_shapes = axes_lengths[: len(axes_lengths) - len(self.added_axes)]
    final_shapes: List[int] = []
    for output_axis, grouping in enumerate(self.output_composite_axes):
        if is_ellipsis_not_in_parenthesis(grouping):
            final_shapes.extend(ellipsis_shape)
        else:
            lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
            final_shapes.append(_product(lengths))
    reduced_axes = self.reduced_elementary_axes
    axes_reordering = self.axes_permutation
    added_axes: Dict[int, int] = {
        pos: axes_lengths[pos_in_elementary]
        for pos, pos_in_elementary in self.added_axes.items()
    }
    # if optimize:
    #     assert len(self.added_axes) == 0
    #     return _optimize_transformation(init_shapes, reduced_axes, axes_reordering, final_shapes)
    return init_shapes, reduced_axes, axes_reordering, added_axes, final_shapes


# Override
einops.einops._reconstruct_from_shape = functools.lru_cache(1024)(
    _reconstruct_from_shape_uncached
)


def infer_unknown_dim(
    in_shape: Sequence[Union[int, sp.Basic]], to_shape: Sequence[Union[int, sp.Basic]]
) -> int:
    assert isinstance(in_shape, Sequence), f"Invalid input shape: {in_shape}"
    assert isinstance(to_shape, Sequence), f"Invalid output shape: {to_shape}"

    num_neg_ones = sum(s == -1 for s in to_shape)
    if num_neg_ones > 1:
        raise ValueError("only one -1 is allowed in to_shape")
    elif num_neg_ones == 1:
        dims_from_left = []
        dims_from_right = []

        # Starting from the left
        for s in to_shape:
            if s == -1:
                break
            dims_from_left.append(s)

        # Starting from the right
        for s in to_shape[::-1]:
            if s == -1:
                break
            dims_from_right.append(s)

        # Compute the missing dimension
        missing_dim = fake_floordiv(
            prod(in_shape), prod(dims_from_left + dims_from_right)
        )

        # Replace -1 with the missing dimension
        shape = dims_from_left + [missing_dim] + list(reversed(dims_from_right))
    else:
        # Remove floor and ceil in in_shape and to_shape
        # TODO(zhanda): here we only do very simple verification
        concrete_in_shape = gsm.subs(in_shape)
        concrete_to_shape = gsm.subs(to_shape)
        assert prod(concrete_in_shape) == prod(
            concrete_to_shape
        ), f"shape does not match size, {concrete_in_shape} -> {concrete_to_shape}"
        shape = to_shape

    return tuple(shape)


def shape_rearrange(in_shape, pattern, axes_lengths):
    assert "->" in pattern, f"Invalid pattern: {pattern}"
    left_str, rght_str = pattern.split("->")
    left = ParsedExpression(left_str)
    rght = ParsedExpression(rght_str)

    # parsing all dimensions to find out lengths
    axis_name2known_length: Dict[Union[str, AnonymousAxis], int] = OrderedDict()

    for elementary_axis, axis_length in axes_lengths:
        axis_name2known_length[elementary_axis] = axis_length

    # first pass: find out lengths of all axes that are not composite
    for i, composite_axis in enumerate(left.composition):
        if len(composite_axis) == 1:
            axis_name2known_length[composite_axis[0]] = in_shape[i]
        else:
            for elementary_axis in composite_axis:
                if elementary_axis not in axis_name2known_length:
                    axis_name2known_length[elementary_axis] = _unknown_axis_length

    # second pass: match all composite axes
    for i, composite_axis in enumerate(left.composition):
        # only composite axes are interesting
        if len(composite_axis) > 1:
            # get lengths list for all axes in composite axis
            composite_axis_lengths = [
                axis_name2known_length[axis] for axis in composite_axis
            ]
            num_unknown_axis = composite_axis_lengths.count(_unknown_axis_length)

            if num_unknown_axis == 0:
                continue
            elif num_unknown_axis == 1:
                # get the only unknown axis
                unknown_axis_idx = composite_axis_lengths.index(_unknown_axis_length)
                # calculate the prod of all known axes
                known_axes_length = prod(
                    composite_axis_lengths[:unknown_axis_idx]
                    + composite_axis_lengths[unknown_axis_idx + 1 :]
                )
                # calculate the length of the unknown axis
                axis_name2known_length[composite_axis[unknown_axis_idx]] = floor_div(
                    in_shape[i], known_axes_length
                )
            else:
                raise ValueError(
                    f"More than one unknown axis in {composite_axis} in {left_str}"
                )

    # create out shape
    out_shape = []
    for elementary_axis in rght.composition:
        out_shape.append(
            prod([axis_name2known_length[axis] for axis in elementary_axis])
        )

    return out_shape
