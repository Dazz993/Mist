# Based on:
# 1. torch/distributed/utils.py
import traceback
from typing import Any, Sequence, Set

import torch
from torch.storage import TypedStorage


def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """This is used as an alternate to ``assert`` when in the backward context
    to print the error message ``s`` since otherwise, it is swallowed."""
    if not cond:
        print(s)
        traceback.print_stack()
        if raise_assertion_error:
            raise AssertionError(s)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> bool:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    with torch.no_grad():
        already_allocated = tensor._typed_storage()._size() == size.numel()
        if not already_allocated:
            tensor_storage_size = tensor._typed_storage()._size()
            _p_assert(
                tensor_storage_size == 0,
                f"Tensor storage should have been resized to be 0 but got {tensor_storage_size}",
            )
            tensor._typed_storage()._resize_(size.numel())
        return not already_allocated


def _free_storage(tensor: torch.Tensor) -> bool:
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    with torch.no_grad():
        already_freed = tensor._typed_storage()._size() == 0
        if not already_freed:
            _p_assert(
                tensor.storage_offset() == 0,
                "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                f"storage offset: {tensor.storage_offset()}\n"
                f"storage size: {tensor._typed_storage()._size()}\n"
                f"tensor shape: {tensor.shape}",
            )
            tensor._typed_storage()._resize_(0)
        return not already_freed


def _resize_storage(tensor: torch.Tensor, numel: int) -> None:
    """
    Resize the underlying storage of ``tensor`` to have ``numel`` elements.

    Args:
        tensor (torch.Tensor): The tensor whose storage to resize.
        numel (int): The number of elements to resize the storage to.
    """
    with torch.no_grad():
        tensor._typed_storage()._resize_(numel)


def _validate_tensor_storage(tensor: torch.Tensor) -> bool:
    """
    Validates that the tensor's storage is contiguous and has the correct size.

    Returns:
        bool: ``True`` if the tensor is valid and ``False`` otherwise.
    """
    with torch.no_grad():
        return tensor._typed_storage()._size() == tensor.numel()


def _inplace_fill_storage(tensor: torch.Tensor, value: torch.Tensor):
    assert tensor.numel() == value.numel()
    with torch.no_grad():
        # ===================================================
        # Original implementation
        # tensor._typed_storage().fill_(value.flatten())
        # ===================================================
        start = tensor.storage_offset()
        end = start + tensor.numel()
        tensor._typed_storage()._setitem(slice(start, end), value)


def _inplace_fill_storage_using_copy(
    tensor: torch.Tensor, value: torch.Tensor, non_blocking: bool = False
):
    assert tensor.numel() == value.numel()
    with torch.no_grad():
        start = tensor.storage_offset()
        end = start + tensor.numel()
        storage = tensor._typed_storage()
        idx = slice(start, end)
        if storage.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8,
            }
            tmp_dtype = interpret_dtypes[storage.dtype]
            tmp_tensor = torch.tensor(
                [], dtype=tmp_dtype, device=storage._untyped_storage.device
            )
            tmp_tensor.set_(
                TypedStorage(
                    wrap_storage=storage._untyped_storage,
                    dtype=tmp_dtype,
                    _internal=True,
                )
            )
        else:
            tmp_tensor = torch.tensor(
                [], dtype=storage.dtype, device=storage._untyped_storage.device
            ).set_(storage)

        tmp_tensor[idx].copy_(value, non_blocking=non_blocking)


def _get_base_tensor_set_with_unique_storage(
    tensors: Sequence[torch.Tensor],
    check_completeness: bool = False,
) -> Set[torch.Tensor]:
    """
    Returns a set of tensors with unique storage.

    Parameters
    ----------
    tensors (Sequence[torch.Tensor]): The sequence of tensors to get the unique
        storage from.

    Returns
    -------
    Set[torch.Tensor]: The set of tensors with unique storage.
    """

    def _get_base_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor if tensor._base is None else tensor._base

    base_tensor_to_tensors = {}

    for tensor in tensors:
        base_tensor = _get_base_tensor(tensor)
        if base_tensor not in base_tensor_to_tensors:
            base_tensor_to_tensors[base_tensor] = []
        base_tensor_to_tensors[base_tensor].append(tensor)

    if check_completeness:
        for base_tensor, sub_tensors in base_tensor_to_tensors.items():
            sub_tensors_set = set()
            sub_tensors_data_ptr_set = set()
            for sub_tensor in sub_tensors:
                if sub_tensor.data_ptr() not in sub_tensors_data_ptr_set:
                    sub_tensors_data_ptr_set.add(sub_tensor.data_ptr())
                    sub_tensors_set.add(sub_tensor)
            assert (
                sum(sub_tensor.numel() for sub_tensor in sub_tensors_set)
                == base_tensor.numel()
            ), (
                "The sub tensors of a base tensor should be complete: "
                f"base_tensor: {base_tensor.shape}\n"
                f"sub_tensors: {[t.shape for t in sub_tensors]}\n"
            )

    ret = set(base_tensor_to_tensors.keys())

    return ret
