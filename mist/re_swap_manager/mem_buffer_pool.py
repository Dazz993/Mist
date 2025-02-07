from typing import List

import torch
import torch.distributed

from mist.logger import get_logger

logger = get_logger()

_GLOBAL_MEMORY_BUFFERS = dict()
_RING_MEMORY_BUFFERS = dict()

def is_first_rank():
    return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0


def allocate_memory_buffer(name, numel, dtype, track_usage, data=None):
    """Allocate a memory buffer."""
    assert (
        name not in _GLOBAL_MEMORY_BUFFERS
    ), f"memory buffer {name} already allocated."
    _GLOBAL_MEMORY_BUFFERS[name] = MemoryBuffer(
        name, numel, dtype, track_usage, data=data
    )
    return _GLOBAL_MEMORY_BUFFERS[name]


def get_memory_buffer(name):
    """Get the memory buffer."""
    return _GLOBAL_MEMORY_BUFFERS[name]

def is_buffer_allocated(name):
    return name in _GLOBAL_MEMORY_BUFFERS

def delete_memory_buffer(name):
    """Delete the memory buffer."""
    del _GLOBAL_MEMORY_BUFFERS[name]


def allocate_ring_memory_buffer(name, num_buffers, numel, dtype, track_usage):
    """Allocate a ring of memory buffers."""
    assert name not in _RING_MEMORY_BUFFERS, f"memory buffer {name} already allocated."
    _RING_MEMORY_BUFFERS[name] = RingMemoryBuffer(
        name, num_buffers, numel, dtype, track_usage
    )
    return _RING_MEMORY_BUFFERS[name]


def get_ring_memory_buffer(name):
    """Get the memory buffer."""
    return _RING_MEMORY_BUFFERS[name]


def delete_ring_memory_buffer(name):
    """Delete the memory buffer."""
    del _RING_MEMORY_BUFFERS[name]


class MemoryBuffer:
    """Contiguous memory buffer.
    Allocate a contiguous memory of type `dtype` and size `numel`. It is
    used to reduce memory fragmentation.

    Usage: After the allocation, the `_start` index is set tot the first
           index of the memory. A memory chunk starting from `_start` index
           can be `allocated` for an input tensor, with the elements of the
           tensor being coppied. The buffer can be reused by resetting the
           `_start` index.
    """

    def __init__(self, name, numel, dtype, track_usage, data=None):
        if is_first_rank():
            element_size = torch.tensor([], dtype=dtype).element_size()
            logger.debug(
                "> building the {} memory buffer with {} num elements "
                "and {} dtype ({:.1f} MB)...".format(
                    name, numel, dtype, numel * element_size / 1024**2
                )
            )
        self.name = name
        self.numel = numel
        self.dtype = dtype
        if data is None:
            self.data = torch.zeros(
                self.numel,
                dtype=self.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            self._start = 0
        else:
            assert data.numel() == self.numel
            assert data.dtype == self.dtype
            assert data.device.index == torch.cuda.current_device()
            self.data = data
            self._start = self.numel

        # Index tracking the start of the free memory.

        # Values used for tracking usage.
        self.track_usage = track_usage
        if self.track_usage:
            self.in_use_value = 0.0
            self.total_value = 0.0

    def reset(self):
        """Reset the buffer start index to the beginning of the buffer."""
        self._start = 0

    def is_in_use(self):
        """Whether the current buffer hold on to any memory."""
        return self._start > 0

    def numel_in_use(self):
        """Return number of elements in use."""
        return self._start

    def add(self, tensor, allow_different_dtype=False):
        """Allocate a chunk of memory from the buffer to tensor and copy
        the values."""
        assert (
            allow_different_dtype or tensor.dtype == self.dtype
        ), f"Input tensor type {tensor.dtype} different from buffer type {self.dtype}"
        # Number of elements of the input tensor.
        tensor_numel = torch.numel(tensor)
        new_start = self._start + tensor_numel
        assert (
            new_start <= self.numel
        ), f"Not enough memory left in the buffer ({tensor_numel} > {self.numel - self._start})"
        # New tensor is a view into the memory.
        new_tensor = self.data[self._start : new_start]
        self._start = new_start
        new_tensor = new_tensor.view(tensor.shape)
        new_tensor.copy_(tensor)
        # Return a pointer to the new tensor.
        return new_tensor

    def new(self, numel):
        """Get a chunk of memory from the buffer."""
        new_start = self._start + numel
        assert (
            new_start <= self.numel
        ), "Not enough memory left in the buffer ({} > {})".format(
            numel, self.numel - self._start
        )
        # New tensor is a view into the memory.
        new_tensor = self.data[self._start : new_start]
        self._start = new_start
        return new_tensor

    def get_data(self):
        """Return the data currently in use."""
        if self.track_usage:
            self.in_use_value += float(self._start)
            self.total_value += float(self.numel)
        return self.data[: self._start]

    def print_average_usage(self):
        """Print memory usage average over time. We would like this value
        to be as high as possible."""
        assert self.track_usage, "You need to enable track usage."
        if is_first_rank():
            print(
                " > usage of {} memory buffer: {:.2f} %".format(
                    self.name, self.in_use_value * 100.0 / self.total_value
                ),
                flush=True,
            )


class RingMemoryBuffer:
    """A ring of memory buffers."""

    def __init__(self, name, num_buffers, numel, dtype, track_usage):
        self.num_buffers = num_buffers
        self.buffers: List[MemoryBuffer] = [
            allocate_memory_buffer(name + f"_{i}", numel, dtype, track_usage)
            for i in range(num_buffers)
        ]
        self._index = -1

    def get_cucr_buffer(self):
        return self.buffers[self._index]

    def get_next_buffer(self):
        for i in range(self.num_buffers):
            self._index += 1
            self._index = self._index % self.num_buffers
            buff = self.buffers[self._index]
            if not buff.is_in_use():
                return buff
        raise RuntimeError("All buffers are in use.")

    def numel(self):
        return sum([buff.numel for buff in self.buffers])
