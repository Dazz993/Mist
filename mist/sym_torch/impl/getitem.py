from numbers import Integral, Number

import sympy as sp
import torch
from sympy import floor

from mist.logger import get_logger
from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op

logger = get_logger()


@register_symbolic_op(torch.Tensor, "__getitem__")
class SymbolicGetitem(SymbolicOp):
    @staticmethod
    def apply(outputs, input, indices):
        input_shape = tuple(input.shape)
        if indices is None:
            # e.g., a[None], which is equivalent to a.unsqueeze(0)
            shape = (1,) + input_shape
        elif isinstance(indices, (Integral, sp.Basic)):
            # e.g., a[1]
            shape = input_shape[1:]
        elif isinstance(indices, slice):
            # e.g., a[1:3]
            # TODO(zhanda): check if the following is correct (for step != 1)
            step = indices.step or 1
            start = indices.start if indices.start is not None else 0
            stop = indices.stop if indices.stop is not None else input_shape[0]
            start = start if start >= 0 else input_shape[0] + start
            stop = stop if stop >= 0 else input_shape[0] + stop
            shape = ((stop - start - 1) // step + 1,) + input_shape[1:]
        elif isinstance(indices, tuple):
            # If there is an Ellipsis, we need to handle it separately.
            # e.g., a[1, ..., 2:3]
            # It must span the entire dimension, so we can't just use
            # the same logic as the following else branch.
            if Ellipsis in indices:
                ellipsis_count = indices.count(Ellipsis)
                if ellipsis_count > 1:
                    raise ValueError("only one ellipsis is allowed")
                ellipsis_index = indices.index(Ellipsis)
                lhs = indices[:ellipsis_index]
                rhs = indices[ellipsis_index + 1 :]
                lhs_valid = [v for v in lhs if v is not None]
                rhs_valid = [v for v in rhs if v is not None]
                ellipsis_len = input.ndim - len(lhs_valid) - len(rhs_valid)

                shape = []
                valid_idx = 0
                for v in lhs:
                    if v is None:
                        shape.append(1)
                    elif isinstance(v, (Integral, sp.Basic)):
                        pass
                        valid_idx += 1
                    elif isinstance(v, (tuple, list)):
                        shape.append(len(v))
                        valid_idx += 1
                    elif isinstance(v, slice):
                        step = v.step or 1
                        start = v.start if v.start is not None else 0
                        stop = v.stop if v.stop is not None else input_shape[valid_idx]
                        start = start if start >= 0 else input_shape[valid_idx] + start
                        stop = stop if stop >= 0 else input_shape[valid_idx] + stop
                        shape.append((stop - start - 1) // step + 1)
                        valid_idx += 1
                    else:
                        raise ValueError(f"unrecognized index: {v}")

                shape.extend(input_shape[valid_idx : valid_idx + ellipsis_len])

                valid_idx += ellipsis_len
                for v in rhs:
                    if v is None:
                        shape.append(1)
                    elif isinstance(v, (Integral, sp.Basic)):
                        valid_idx += 1
                    elif isinstance(v, (tuple, list)):
                        shape.append(len(v))
                        valid_idx += 1
                    elif isinstance(v, slice):
                        step = v.step or 1
                        start = v.start if v.start is not None else 0
                        stop = v.stop if v.stop is not None else input_shape[valid_idx]
                        start = start if start >= 0 else input_shape[valid_idx] + start
                        stop = stop if stop >= 0 else input_shape[valid_idx] + stop
                        shape.append((stop - start - 1) // step + 1)
                        valid_idx += 1

            else:
                # e.g., a[1, 2:3]
                # Need to consider unspecified dimensions.
                shape = []
                valid_idx = 0

                special_path = False
                contained_torch_tensor = []
                for v in indices:
                    if isinstance(v, torch.Tensor):
                        contained_torch_tensor.append(v)
                if contained_torch_tensor:
                    assert (v.shape == contained_torch_tensor[0].shape for v in contained_torch_tensor), (
                        f"Different torch tensors are not supported yet while it is actually doable."
                    )
                    shape = contained_torch_tensor[0].shape
                    special_path = True

                if not special_path:
                    for v in indices:
                        if v is None:
                            shape.append(1)
                        elif isinstance(v, (Integral, sp.Basic)):
                            valid_idx += 1
                        elif isinstance(v, (tuple, list)):
                            shape.append(len(v))
                            valid_idx += 1
                        elif isinstance(v, slice):
                            step = v.step or 1
                            start = v.start if v.start is not None else 0
                            stop = v.stop if v.stop is not None else input_shape[valid_idx]
                            start = start if start >= 0 else input_shape[valid_idx] + start
                            stop = stop if stop >= 0 else input_shape[valid_idx] + stop
                            shape.append((stop - start - 1) // step + 1)
                            valid_idx += 1
                        else:
                            raise ValueError(f"unrecognized index: {v}")

                    shape += input_shape[valid_idx:]

        elif isinstance(indices, torch.Tensor):
            shape = indices.shape
            shape += input_shape[1:]
        else:
            raise ValueError(f"unrecognized indices: {indices}")

        ret = SymbolicTensor(outputs, tuple(shape))
        ret.context = SymbolicOpContext(op=SymbolicGetitem, saved_tensors=[input])

        # TODO(zhanda): remove this in the future. Leave it here for debugging.
        # logger.debug(
        #     f"SymbolicGetitem: {input_shape}[{indices}] -> {ret.shape}, {ret.concrete_shape}"
        # )

        return ret


@register_symbolic_op(torch.Tensor, "__setitem__")
class SymbolicSetitem(SymbolicOp):
    @staticmethod
    def apply(outputs, input, indices, value):
        assert isinstance(input, torch.Tensor)
        assert isinstance(value, (torch.Tensor, Number)), f"{value}"
        return None
