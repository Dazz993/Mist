import time
from typing import Callable, Sequence, Union, Optional, Tuple, List, Dict, Any, Iterable
from enum import Enum, auto

from mist.utils.memory import cuda_empty_cache
import numpy as np
import torch
from torch.utils.weak import TensorWeakRef

from mist.logger import get_logger
from mist.utils.device import get_device
from mist.utils.storage import (
    _resize_storage,
    _inplace_fill_storage,
    _inplace_fill_storage_using_copy,
    _get_base_tensor_set_with_unique_storage,
)

logger = get_logger()

# DEFAULT_MEMORY_LIMIT = 2 * 1024 * 1024  # 2MB
DEFAULT_MEMORY_LIMIT = 0


def get_swapped(
    tensors: Iterable[torch.Tensor],
    swapped_ratio: Optional[float] = None,
    swapped_size: Optional[int] = None,
    memory_limit: Optional[int] = DEFAULT_MEMORY_LIMIT,
) -> List[Tuple[torch.Tensor, Optional[int]]]:
    if swapped_ratio is None and swapped_size is None:
        raise ValueError("Either swapped_ratio or swapped_size should be provided.")

    tensors = [t for t in tensors if t is not None]
    tensors = list(
        _get_base_tensor_set_with_unique_storage(tensors, check_completeness=True)
    )
    # For swapped tensors, we need more careful management and assumption.
    for tensor in tensors:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.numel() == tensor._typed_storage()._size(), (
            f"tensor.numel(): {tensor.numel()}, "
            f"tensor._typed_storage()._size(): {tensor._typed_storage()._size()}"
        )
        assert (
            tensor.nbytes == tensor._typed_storage()._size() * tensor.element_size()
        ), (
            f"tensor.nbytes: {tensor.nbytes}, "
            f"tensor._typed_storage()._size() * tensor.element_size(): {tensor._typed_storage()._size() * tensor.element_size()}"
        )
        if hasattr(tensor, "_is_fully_cuda") or hasattr(tensor, "_is_fully_cpu"):
            assert getattr(tensor, "_is_fully_cuda", True) and getattr(
                tensor, "_is_fully_cpu", True
            ), (
                f"tensor is not fully in cuda or cpu memory, thus the numel may not be correct."
                f"tensor._is_fully_cuda: {getattr(tensor, '_is_fully_cuda', None)}, "
                f"tensor._is_fully_cpu: {getattr(tensor, '_is_fully_cpu', None)}"
            )

    all_tensor_size = sum(t.nbytes for t in set(tensors))
    if swapped_ratio is not None:
        swapped_size = int(all_tensor_size * swapped_ratio)

    swappable = list(set([t for t in tensors if t.nbytes >= memory_limit]))
    swappable.sort(key=lambda x: x.nbytes, reverse=True)

    swapped = []
    to_be_swapped_size = swapped_size
    for tensor in swappable:
        if to_be_swapped_size < memory_limit:
            break
        elif to_be_swapped_size <= tensor.nbytes:  # [memory_limit, tensor.nbytes]
            remained_elements_to_swap = to_be_swapped_size // tensor.element_size()
            numel_in_cuda = tensor.numel() - remained_elements_to_swap
            swapped.append((tensor, numel_in_cuda))
            break
        swapped.append((tensor, 0))
        to_be_swapped_size -= tensor.nbytes

    return swapped


def is_tensor_on_expected_device(tensor: torch.Tensor, expected_device: str = "cuda"):
    assert expected_device in ("cuda", "cpu")
    if hasattr(tensor, "_is_fully_cuda") or hasattr(tensor, "_is_fully_cpu"):
        if expected_device == "cuda":
            return tensor._is_fully_cuda
        else:
            return tensor._is_fully_cpu
    else:
        return tensor.device.type == expected_device


def init_tensor_swap_handle(
    tensor: torch.Tensor,
    dst_numel_in_cuda_for_partial: int,
    cuda_device=None,
):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"tensor must be a torch.Tensor, but got {type(tensor)}")

    if getattr(tensor, "_swap_handle", None) is not None:
        swap_handle = tensor._swap_handle
        tensor._dst_numel_in_cuda_for_partial = dst_numel_in_cuda_for_partial
    else:
        # During initialization, swap_handle is registered to the tensor
        swap_handle = TensorSwapHandle(
            tensor, cuda_device, dst_numel_in_cuda_for_partial
        )
    return swap_handle


def swap_(
    tensor: torch.Tensor,
    state: str = None,
    dst_numel_in_cuda_for_partial: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
    cache_cpu_data=False,
    cuda_buffer: Optional[torch.Tensor] = None,
    cpu_buffer: Optional[torch.Tensor] = None,
) -> bool:
    # Update the state and dst_numel_in_cuda_for_partial
    if state not in ("cpu", "cuda", "partial", None):
        raise ValueError(
            f"state must be cpu, cuda or partial, or None, but got {state}"
        )
    state = state or "partial"
    if state == "partial" and dst_numel_in_cuda_for_partial is None:
        raise ValueError(
            "dst_numel_in_cuda_for_partial must be provided when state is None"
        )

    # Update the the cuda stream
    if stream is None:
        stream = torch.cuda.current_stream()

    # Get the real dst_numel_in_cuda
    swap_handle = getattr(tensor, "_swap_handle", None)
    if state == "cpu":
        dst_numel_in_cuda = 0
    elif state == "cuda":
        # In any case, we should use the numel of the tensor
        dst_numel_in_cuda = tensor.numel()
    else:  # state == "partial"
        dst_numel_in_cuda = dst_numel_in_cuda_for_partial

    if swap_handle is None and dst_numel_in_cuda >= tensor.numel():
        return False
    elif swap_handle is None:  # dst_numel_in_cuda < tensor.numel()
        swap_handle = init_tensor_swap_handle(
            tensor,
            dst_numel_in_cuda_for_partial=dst_numel_in_cuda_for_partial,
            cuda_device=get_device(torch.cuda.current_device()),
        )
    else:
        swap_handle = tensor._swap_handle

    worked = False
    if swap_handle.numel_in_cuda() != dst_numel_in_cuda:
        swap_handle.swap(
            dst_numel_in_cuda=dst_numel_in_cuda,
            copy_stream=stream,
            cache_cpu_data=cache_cpu_data,
            cuda_buffer=cuda_buffer,
            cpu_buffer=cpu_buffer,
        )
        worked = True

    return worked


class TensorSwapHandle:
    """
    A tensor_swap_handle can be in one of three states:
    1. Fully in cuda memory. In this case, `data` is a torch.Tensor.
    2. Fully in cpu memory. In this case, `data` is a torch.Tensor.
    3. Partially in cuda memory. In this case, `data` is a tuple of two torch.Tensors
         (the first tensor is in cuda memory, the second tensor is in cpu memory)
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        cuda_device=None,
        dst_numel_in_cuda_for_partial: Optional[int] = None,
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"tensor must be a torch.Tensor, but got {type(tensor)}")

        if getattr(tensor, "_swap_handle", None) is not None:
            raise ValueError("tensor already has a swap handle")
        if getattr(tensor, "_cuda_data", None) is not None:
            raise ValueError("tensor already has a swap handle")
        if getattr(tensor, "_cpu_data", None) is not None:
            raise ValueError("tensor already has a swap handle")

        # The tensor_weakref is used to avoid circular reference
        self.tensor_weakref = TensorWeakRef(tensor)
        # Get the device of the tensor
        self.cuda_device = cuda_device or (
            tensor.device if tensor.is_cuda else get_device(torch.cuda.current_device())
        )
        # The number of elements in cuda memory when the tensor is partially in cuda memory
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self._full_numel = None
        self._empty_cuda_data = torch.zeros(
            [], dtype=self.dtype, device=self.cuda_device
        )

        if tensor.is_cuda:
            tensor._cuda_data = tensor.data.view(-1)
            tensor._cpu_data = torch.zeros(
                0, dtype=self.dtype, device="cpu", pin_memory=True
            )
        elif tensor.is_cpu:
            tensor._cuda_data = torch.zeros(
                0, dtype=self.dtype, device=self.cuda_device
            )
            tensor._cpu_data = tensor.data.view(-1)
        else:
            raise ValueError(
                f"tensor must be either in cuda or cpu memory, but got {tensor.device.type}"
            )

        # Register the swap handle to the tensor
        tensor._swap_handle = self
        tensor._dst_numel_in_cuda_for_partial = dst_numel_in_cuda_for_partial
        tensor._full_numel = tensor.numel()

    def healthy(self):
        return self.numel_in_cuda() == self.numel()

    def numel(self):
        if self._full_numel is None:
            self._full_numel = np.prod(self.shape)
        return self._full_numel

    def numel_in_cuda(self):
        tensor = self.tensor_weakref()
        return tensor._cuda_data.numel() if tensor._cuda_data is not None else 0

    def numel_in_cpu_including_cached(self):
        tensor = self.tensor_weakref()
        return tensor._cpu_data.numel() if tensor._cpu_data is not None else 0

    def numel_in_cpu_excluding_cached(self):
        return self.numel() - self.numel_in_cuda()

    def numel_in_cpu_cached(self):
        return (
            self.numel_in_cpu_including_cached() - self.numel_in_cpu_excluding_cached()
        )
    
    # TODO(tmp)
    @torch.no_grad()
    def swap(
        self,
        dst_numel_in_cuda: int,
        copy_stream: torch.cuda.Stream,
        cache_cpu_data: bool = True,
        cuda_buffer: Optional[torch.Tensor] = None,
        cpu_buffer: Optional[torch.Tensor] = None,
        non_blocking: bool = True,
    ):
        with torch.cuda.stream(copy_stream):
            self._swap(
                dst_numel_in_cuda,
                copy_stream,
                cache_cpu_data=cache_cpu_data,
                cuda_buffer=cuda_buffer,
                cpu_buffer=cpu_buffer,
                non_blocking=non_blocking,
            )

    @torch.no_grad()
    def _swap(
        self,
        dst_numel_in_cuda: int,
        copy_stream: torch.cuda.Stream,
        cache_cpu_data: bool = True,
        cuda_buffer: Optional[torch.Tensor] = None,
        cpu_buffer: Optional[torch.Tensor] = None,
        non_blocking: bool = True,
    ):
        """This method moves the last `num_elements_to_move` elements of the tensor to `dst_device`.

        Parameters
        ----------
        dst_numel_in_cuda
            destination number of elements in cuda memory
        deallocate_unncessary_cpu_data, optional
            whether deallocate unnecessary cpu data, by default False
        overwrite_cpu_data, optional
            whether overwrite cpu data, by default False. This will only be used when the tensor
            is fully in cuda memory and is trying to move some elements from cuda to cpu.
        """
        deallocate_unncessary_cpu_data = not cache_cpu_data
        use_cached_cpu_data = cache_cpu_data

        prev_alloc = torch.cuda.memory_allocated() / 1024**2

        # The number of elements that will be exclusively in cpu memory
        dst_numel_in_cpu_excluding_cached = self.numel() - dst_numel_in_cuda
        # The number of elements that are in gpu memory
        src_numel_in_cuda = self.numel_in_cuda()
        # The number of elements that are exclusively in cpu memory
        src_numel_in_cpu_excluding_cached = self.numel_in_cpu_excluding_cached()
        # The number of elements that are in cpu memory
        src_numel_in_cpu_including_cached = self.numel_in_cpu_including_cached()

        tensor = self.tensor_weakref()
        input_device = tensor.device
        input_storage_size = tensor.data._typed_storage()._size()

        if src_numel_in_cuda == dst_numel_in_cuda:
            return

        if cuda_buffer is not None:
            assert cuda_buffer.numel() == dst_numel_in_cuda, (
                f"tensor.numel(): {tensor.numel()}, "
                f"cuda_buffer.numel(): {cuda_buffer.numel()}, "
                f"dst_numel_in_cuda: {dst_numel_in_cuda}"
            )
            assert cuda_buffer.device == self.cuda_device

        if cpu_buffer is not None:
            assert (
                not use_cached_cpu_data
            ), "cpu_buffer is not used when use_cached_cpu_data is True"
            assert cpu_buffer.numel() == dst_numel_in_cpu_excluding_cached, (
                f"tensor.numel(): {tensor.numel()}, "
                f"cpu_buffer.numel(): {cpu_buffer.numel()}, "
                f"dst_numel_in_cpu_excluding_cached: {dst_numel_in_cpu_excluding_cached}"
            )

        # Move some elements from cuda to cpu. should consider the cache.
        if src_numel_in_cuda > dst_numel_in_cuda:
            cuda_delta = src_numel_in_cuda - dst_numel_in_cuda
            cpu_delta_including_cached = (
                dst_numel_in_cpu_excluding_cached - src_numel_in_cpu_including_cached
            )
            cpu_delta_excluding_cached = (
                dst_numel_in_cpu_excluding_cached - src_numel_in_cpu_excluding_cached
            )
            if use_cached_cpu_data and cpu_delta_including_cached <= 0:
                # All the data that should be in the cpu is already cached
                # we don't need to do anything since cpu has all it needs
                pass
            elif cpu_buffer is None:
                # Some data that should be in the cpu is not cached
                # we ignore the partial cache
                # TODO(zhanda): see whether we should consider the partial cache here
                dst_cpu_data = torch.empty(
                    dst_numel_in_cpu_excluding_cached,
                    dtype=self.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                # with torch.cuda.stream(copy_stream):
                # tmp_data = tensor._cuda_data[-cpu_delta_excluding_cached:].detach().clone()
                dst_cpu_data.view(-1)[:cpu_delta_excluding_cached].copy_(
                    tensor._cuda_data[
                        tensor._cuda_data.numel() - cpu_delta_excluding_cached :
                    ],
                    non_blocking=non_blocking,
                )
                dst_cpu_data.view(-1)[cpu_delta_excluding_cached:].copy_(
                    tensor._cpu_data[
                        tensor._cpu_data.numel() - src_numel_in_cpu_excluding_cached :
                    ],
                    non_blocking=non_blocking,
                )
                tensor._cpu_data = dst_cpu_data
            else:  # cpu_buffer is not None
                cpu_buffer.view(-1)[:cpu_delta_excluding_cached].copy_(
                    tensor._cuda_data[
                        tensor._cuda_data.numel() - cpu_delta_excluding_cached :
                    ],
                    non_blocking=non_blocking,
                )
                cpu_buffer.view(-1)[cpu_delta_excluding_cached:].copy_(
                    tensor._cpu_data[
                        tensor._cpu_data.numel() - src_numel_in_cpu_excluding_cached :
                    ],
                    non_blocking=non_blocking,
                )
                tensor._cpu_data = cpu_buffer

            if cuda_buffer is None:
                tensor._cuda_data.record_stream(copy_stream)
                tensor._cuda_data = tensor._cuda_data[:-cuda_delta].detach().clone()
                # tensor._cuda_data._typed_storage()._resize_(tensor._cuda_data.numel())
            else:
                # with torch.cuda.stream(copy_stream):
                cuda_buffer.view(-1).copy_(
                    tensor._cuda_data[:-cuda_delta], non_blocking=non_blocking
                )
                tensor._cuda_data.record_stream(copy_stream)
                tensor._cuda_data = cuda_buffer

        # Move some elements from cpu to cuda. should consider the deallocation.
        else:
            cuda_delta = dst_numel_in_cuda - src_numel_in_cuda

            if cuda_buffer is None and dst_numel_in_cuda == self.numel():
                _resize_storage(tensor.data, dst_numel_in_cuda)
                dst_cuda_data = tensor.data.view(-1)
            elif cuda_buffer is None and dst_numel_in_cuda < self.numel():
                dst_cuda_data = torch.empty(
                    dst_numel_in_cuda,
                    dtype=self.dtype,
                    device=self.cuda_device,
                )
            else:
                dst_cuda_data = cuda_buffer.view(-1)
            dst_cuda_data.view(-1)[:src_numel_in_cuda].copy_(
                tensor._cuda_data, non_blocking=non_blocking
            )
            dst_cuda_data.view(-1)[src_numel_in_cuda:].copy_(
                tensor._cpu_data[:cuda_delta], non_blocking=non_blocking
            )
            tensor._cuda_data.record_stream(copy_stream)
            tensor._cuda_data.data.record_stream(copy_stream)
            tensor._cuda_data = dst_cuda_data
            if deallocate_unncessary_cpu_data:
                tensor._cpu_data = tensor._cpu_data[cuda_delta:].detach().clone()

        # Fully CUDA
        if dst_numel_in_cuda == self.numel():
            # ==========================================================
            # Deprecated:
            # If a.data = b, and then we change b.data, a.data won't
            # change accordingly. Thus, we use torch.Storage to
            # manipulate the underlying storage.
            # ==========================================================
            # Note: be careful that if tensor.data is on cpu, then
            # this may not work. We may still need to allocate a new
            # empty tensor and then fill it with the data.
            # _resize_storage(tensor.data, self.numel())
            # _inplace_fill_storage(tensor.data, tensor._cuda_data)
            # ==========================================================
            # tensor.data = tensor._cuda_data.view(self.shape)
            # ==========================================================
            # TODO(zhanda): add more tests to test the correctness for
            # activation swapping.
            # if cuda_buffer is None:
            #     # This is mainly for activation swapping for now.
            #     _resize_storage(tensor.data, self.numel())
            #     _inplace_fill_storage(tensor.data, tensor._cuda_data)
            # else:
            # _resize_storage(tensor.data, self.numel())
            if cuda_buffer is not None:
                tensor.data.record_stream(copy_stream)
                tensor.data = dst_cuda_data.view(self.shape)
                dst_cuda_data.record_stream(copy_stream)
            else:
                # tensor[:] = dst_cuda_data.view(self.shape)
                pass
            # tensor[:] = tensor._cuda_data.view(self.shape)
            tensor.data = tensor.data.view(self.shape)
            tensor._cuda_data = tensor.data.view(-1)
            tensor._is_fully_cuda = True

        # Fully CPU
        elif dst_numel_in_cuda == 0:
            # deallocate the cuda data
            if cuda_buffer is None:
                _resize_storage(tensor.data, 0)
            # else:
                # tensor.data = tensor._cpu_data.view(self.shape)
                # tensor.data = self._empty_cuda_data
            tensor._is_fully_cuda = False

        # Partial swap, we let the tensor to be non-complete
        else:
            # deallocate the tensor data
            tensor.data.record_stream(copy_stream)
            if cuda_buffer is None:
                _resize_storage(tensor.data, 0)
            # tensor.data = torch.empty(0, dtype=self.dtype, device=self.cuda_device)
            tensor._is_fully_cuda = False

        # tensor._cuda_data.record_stream(copy_stream)
        # tensor.data.record_stream(copy_stream)
        # tensor.record_stream(copy_stream)

        tensor.record_stream(copy_stream)
        tensor.data.record_stream(copy_stream)
        if tensor._cuda_data is not None:
            tensor._cuda_data.record_stream(copy_stream)

        logger.debug(
            f"[Swapping Tensor]: [Shape]: {tuple(self.shape)}, [Dtype]: {self.dtype}. "
            f"[Src Numel In CUDA]: {src_numel_in_cuda}, [Dst Numel In CUDA]: {dst_numel_in_cuda}, "
            # f"[Input Storage]: {input_storage_size}, [Input Device]: {input_device}, "
            f"[Output Storage] {tensor._typed_storage()._size()}, [Output Device]: {tensor.device}, "
            f"[Prev Alloc]: {prev_alloc:.2f} MB, "
            f"[Curr Alloc]: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB"
        )

        return

    @staticmethod
    def clean(tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"tensor must be a torch.Tensor, but got {type(tensor)}")
        if getattr(tensor, "_swap_handle", None) is not None:
            delattr(tensor, "_swap_handle")
        if getattr(tensor, "_cuda_data", None) is not None:
            delattr(tensor, "_cuda_data")
        if getattr(tensor, "_cpu_data", None) is not None:
            delattr(tensor, "_cpu_data")


def preprocess_tensors_to_be_swapped(tensors: Sequence[torch.Tensor]):
    """Not all tensors can be swapped. This function checks whether the tensors
    can be swapped and preprocesses the tensors to be swapped.

    # TODO(zhanda): Improve the logic to support more general cases.
    Tensors follow the following rule can be swapped:
    - All usage of tensor's storage in the tensor sequence.

    However, it's hard to check. So we simplify the rule to:
    - For all storage_data_ptr and storage_size,
        sum(t.numel() for t with different data_ptr) == storage_size

    Parameters
    ----------
    tensors
        a sequence of tensors to be swapped.
    """
    storage_data_ptrs: Dict[int, List[torch.Tensor]] = {}
    tensors_to_be_swapped = []

    for tensor in tensors:
        # Add the tensor to corresponding storage_data_ptr.
        storage_data_ptr = tensor._typed_storage()._data_ptr()
        storage_data_ptrs.setdefault(storage_data_ptr, []).append(tensor)

    for storage_data_ptr, tensors in storage_data_ptrs.items():
        storage_size = tensors[0]._typed_storage()._size()
        data_ptr_memo = set()
        unique_tensors = []
        for tensor in tensors:
            data_ptr = tensor.data_ptr()
            if data_ptr in data_ptr_memo:
                continue
            data_ptr_memo.add(data_ptr)
            unique_tensors.append(tensor)
        storage_fullfilled = sum(t.numel() for t in unique_tensors) == storage_size

        if storage_fullfilled:
            dtype = unique_tensors[0].dtype
            device = unique_tensors[0].device
            tmp_tensor = torch.tensor([], dtype=dtype, device=device).set_(
                unique_tensors[0]._typed_storage()
            )
            tensors_to_be_swapped.append(tmp_tensor)
        else:
            raise ValueError(
                f"Storage with data_ptr {storage_data_ptr} does not fullfill the swapping condition."
                f"storage_size: {storage_size}, "
                f"sum of numel: {sum(t.numel() for t in unique_tensors)}"
            )

    return tensors_to_be_swapped


def collect_stats_for_maybe_swapped_tensor(tensor: torch.Tensor, fn: Callable):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"tensor must be a torch.Tensor, but got {type(tensor)}")
    
    swap_handle = getattr(tensor, "_swap_handle", None)

    if swap_handle is None:
        return fn(tensor)

    numel_in_cuda = swap_handle.numel_in_cuda()
    numel_in_cpu_including_cached = swap_handle.numel_in_cpu_including_cached()
    assert numel_in_cuda + numel_in_cpu_including_cached == tensor.numel(), (
        f"numel_in_cuda: {numel_in_cuda}, "
        f"numel_in_cpu_including_cached: {numel_in_cpu_including_cached}, "
        f"tensor.numel(): {tensor.numel()}"
    )

    full_tensor = torch.empty_like(tensor, device=tensor.device)
    if numel_in_cuda > 0:
        full_tensor[:numel_in_cuda].copy_(tensor._cuda_data)
    if numel_in_cpu_including_cached > 0:
        full_tensor[numel_in_cuda:].copy_(tensor._cpu_data)

    # cuda_empty_cache()
    
    return fn(full_tensor)