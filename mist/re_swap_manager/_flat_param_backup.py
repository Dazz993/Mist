import argparse
from typing import List, Optional, Union, Tuple, Any, Dict, Callable, Sequence
from enum import Enum, auto
import functools

import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
from torch.distributed import ProcessGroup
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import flat_param as torch_fsdp_flat_param
from torch.distributed.fsdp._common_utils import _no_dispatch_record_stream
from torch.distributed.fsdp._runtime_utils import (
    _div_if_needed,
    _check_grad_to_accumulate,
)
from torch.distributed.utils import _p_assert

from mist.logger import get_logger, update_logger_formatter_for_rank
from mist.re_swap_manager.mem_buffer_pool import (
    get_memory_buffer,
    get_ring_memory_buffer,
    MemoryBuffer,
    RingMemoryBuffer,
)
from mist.re_swap_manager.swap_backend import swap_
from mist.re_swap_manager.mem_buffer_pool import allocate_memory_buffer

logger = get_logger(__name__)

"""
We subclass the FlatParamHandle to override most of the methods.

==================================================================================
# NOTE(zhanda-2023/12/09): **Deprecated** because we find ways to let the grads of the
# params to be accumulated before the full_backward_hook is called. That is, we
# reassign the params inside the forward pass. This will make sure the grad_fn of
# the Slice/Alias/View's _sequence_nr larger than BackwardHook's _sequence_nr. 
For parameters:
1. flat_param will be created and be used as the root data.
2. flat_param either points to _full_param_padded or _local_shard.
3. _full_param_padded can be freed
4. _local_shard will always be maintained as the local shard of the flat_param.
5. _params will be empty when being sharded. and after unsharding, its data will be
   some part of the flat_param.

for grads:
1. grads will be accumulated to param.grad for param in _params because these are 
   the variables that are used in the forward pass. The reason that we don't let 
   the params to be view of the flat param is that if we do so, it's quite possible
   that the post backward hook is fired before the accumu grad_fn of the params are 
   called. This is okay in FSDP setting, but since we want the perfect overlapping and
   static planning and tuning, this is not okay for us.
2. however, after grads are accumulated (which means in the full_backward_hook),
   the grads will be copied to the flat_grad, `_temp_flat_grad`.
3. reduce_scatter will be called on the `_temp_flat_grad`.
4. new grads will be accumulated to the _saved_grad_shard.
5. all other grads (flat_param.grad, and param.grad for param in _params) will be
   deallocated.
==================================================================================

**UPDATED**:
For parameters:
1. flat_param will be created and be used as the root data.
2. flat_param either points to _full_param_padded or _local_shard.
3. _full_param_padded can be freed. During unsharding, if the buffer is inputted,
   the buffer will be temporarily used as the storage of the unsharded flat_param.
4. _local_shard will always be maintained as the local shard of the flat_param.
   If it has been swapped out, then there will be a buffer to store the cuda data.
5. after FlatParam/Handle is initialized, the module.parameters() will be empty.
   params is still in the original place but as a tensor instead of nn.Parameter.
   This is to say, the params are not in `module._parameters` anymore.
   
For grads:
1. grads will be accumulated to param.grad (and then accumulated to flat_param.grad)
   because params will be view of the flat_param. If the buffer is inputted,
   the buffer will be temporarily used as the storage of the grads.
2. however, after grads are accumulated (which means in the full_backward_hook),
   the grads will be copied to the flat_grad, `_temp_flat_grad`.
3. reduce_scatter will be called on the `_temp_flat_grad`.
4. new grads will be accumulated to the _saved_grad_shard.
5. all other grads (flat_param.grad, and param.grad for param in _params) will be
   deallocated.
"""

DTYPE_RGRAD = Tuple[torch.dtype, bool]


class HandleTrainingState(Enum):
    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


class HandleShardingStrategy(Enum):
    NO_SHARD = auto()
    OPT_ONLY = ZeRO_1 = auto()
    OPT_AND_GRAD = ZeRO_2 = auto()
    FULL_SHARD = ZeRO_3 = auto()

    # Given shard_weights, shard_grads, and shard_opts, return the sharding strategy.
    def from_shard_flags(
        shard_weights: bool, shard_grads: bool, shard_opts: bool
    ) -> "HandleShardingStrategy":
        """Return the sharding strategy given the shard flags"""
        if shard_opts and shard_grads and shard_weights:
            return HandleShardingStrategy.FULL_SHARD
        elif shard_opts and shard_grads:
            return HandleShardingStrategy.OPT_AND_GRAD
        elif shard_opts:
            return HandleShardingStrategy.OPT_ONLY
        else:
            return HandleShardingStrategy.NO_SHARD


class GradReduceOp(Enum):
    NOOP = auto()
    ALL_REDUCE = auto()
    REDUCE_SCATTER = auto()


GRAD_SHARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.FULL_SHARD,
    HandleShardingStrategy.OPT_AND_GRAD,
)


class FlatParamHandle(torch_fsdp_flat_param.FlatParamHandle):
    def __init__(
        self,
        params: Sequence[Union[nn.Parameter, torch.Tensor]],
        module: nn.Module,
        device: torch.device,
        sharding_strategy: HandleShardingStrategy,
        process_group: dist.ProcessGroup,
        all_gather_process_group: Optional[dist.ProcessGroup] = None,
        reduce_scatter_process_group: Optional[dist.ProcessGroup] = None,
        dst_numel_in_cuda_for_partial_weights: Optional[int] = None,
        dst_numel_in_cuda_for_partial_grads: Optional[int] = None,
    ):
        self.all_gather_process_group = all_gather_process_group or process_group
        self.reduce_scatter_process_group = (
            reduce_scatter_process_group or process_group
        )
        super().__init__(
            params=params,
            fully_sharded_module=module,
            device=device,
            sharding_strategy=sharding_strategy,
            offload_params=False,
            mp_param_dtype=None,
            mp_reduce_dtype=None,
            keep_low_precision_grads=False,
            process_group=self.all_gather_process_group,
            use_orig_params=False,
        )
        self._module = self._fully_sharded_module
        self._gradient_predivide_factor = (
            default_hooks.DefaultState._get_gradient_predivide_factor(self.world_size)
        )
        self._gradient_postdivide_factor = (
            self.world_size / self._gradient_predivide_factor
        )

        # ###########
        # Post init #
        # ###########
        # NOTE: self.shard() will partition the params as long as the one of the
        # shardings is enabled.
        self.shard()
        self.init_flat_param_attributes()
        flat_param = self.flat_param
        if not self.uses_sharded_strategy:
            flat_param._full_param_padded = flat_param.data
            flat_param._padded_unsharded_size = flat_param._unpadded_unsharded_size
        # _full numel is the full numel of the unsharded flat param.
        self._full_numel = flat_param._unpadded_unsharded_size.numel()
        # _sharded_numel is the numel of the sharded flat param.
        self._sharded_numel = flat_param.numel()
        # _sharded_numel is full or sharded depending on the sharding strategy.
        self._sharded_weight_numel = (
            self._sharded_numel
            if self.uses_weight_sharding_strategy
            else self._full_numel
        )
        self._sharded_grad_numel = (
            self._sharded_numel
            if self.uses_grad_sharding_strategy
            else self._full_numel
        )
        self._dst_numel_in_cuda_for_partial_weights = (
            self._sharded_weight_numel
            if dst_numel_in_cuda_for_partial_weights is None
            else dst_numel_in_cuda_for_partial_weights
        )
        self._dst_numel_in_cuda_for_partial_grads = (
            self._sharded_grad_numel
            if dst_numel_in_cuda_for_partial_grads is None
            else dst_numel_in_cuda_for_partial_grads
        )

        # Buffer
        self._use_memory_buffer = True
        self._partial_weight_buffer: Optional[MemoryBuffer] = None
        self._partial_grad_buffer: Optional[MemoryBuffer] = None
        self._curr_occupied_full_weights_buffer: Optional[MemoryBuffer] = None
        self._curr_occupied_full_grads_buffer: Optional[MemoryBuffer] = None
        if self._use_memory_buffer:
            self._create_partial_memory_buffers()
            logger.debug(f"Created partial memory buffers for {self._module.name}")

        # Swap out the weights
        # If weight sharding strategy is not used, then we should recover the full
        # weights after the init, because we will always call the shard() method.
        if not self.uses_weight_sharding_strategy:
            self.alloc_full_weights()
            self.unshard()
            self.flat_param._local_shard = self.flat_param.data
            self.swap_out_weights()
            self.dealloc_full_weights()
        else:
            self.swap_out_weights()

    # def _use_unsharded_views(self, as_params: bool) -> None:
    #     """
    #     Override to make sure:
    #     1. `as_params` is always False, which makes sure that the params are always
    #        real `nn.Parameter`s. This is to ensure the grads of params are calculated
    #        before the fire of the full_backward_hook.
    #     2. TODO(zhanda): check whether it works for AC (activation checkpointing)

    #     NOTE(zhanda): this method is called during the `super().__init__()` call.
    #     """
    #     super()._use_unsharded_views(as_params=True)

    def _create_partial_memory_buffers(self):
        """
        Create the partial memory buffers. If the dst_numel_in_cuda is the same as the
        current numel, then directly reuse the data for the memory buffer.
        """
        partial_weight_numel = self._dst_numel_in_cuda_for_partial_weights
        data = (
            self.flat_param if partial_weight_numel == self.flat_param.numel() else None
        )
        self._partial_weight_buffer: MemoryBuffer = allocate_memory_buffer(
            name=f"{self._module.name}.partial_weight_buffer_dtype_{self.flat_param.dtype}_requires_grad_{self.flat_param.requires_grad}",
            numel=partial_weight_numel,
            dtype=self.flat_param.dtype,
            track_usage=True,
            data=data,
        )
        if self.flat_param.requires_grad:
            partial_grad_numel = self._dst_numel_in_cuda_for_partial_grads
            self._partial_grad_buffer: MemoryBuffer = allocate_memory_buffer(
                name=f"{self._module.name}.partial_grad_buffer_dtype_{self.flat_param.dtype}",
                numel=partial_grad_numel,
                dtype=self.flat_param.dtype,
                track_usage=True,
            )

    def set_training_state(self, state: HandleTrainingState):
        self._training_state = state

    def is_weight_unsharded(self) -> bool:
        return self.flat_param.numel() == self._full_numel

    def is_grad_unsharded(self) -> bool:
        if self.flat_param.grad is None:
            return False
        return self.flat_param.grad.numel() == self._full_numel

    def needs_full_weights_allocation(self) -> bool:
        if self.flat_param.numel() > self._full_numel:
            raise ValueError(
                f"FlatParam numel: {self.flat_param.numel()}, "
                f"Full numel: {self._full_numel}."
            )
        elif self.flat_param.numel() == self._full_numel:
            return False
        # If the weight is not sharded and the swapping is disabled, then we don't
        # need to allocate the full weights.
        if (
            self._sharding_strategy != HandleShardingStrategy.FULL_SHARD
            and self._dst_numel_in_cuda_for_partial_weights == self.flat_param.numel()
        ):
            raise ValueError(
                "This should not happen. Check the logic and add the comment here."
            )
            return False
        return True

    def needs_full_grads_allocation(self) -> bool:
        if self.flat_param.grad is not None:
            if self.flat_param.grad.numel() > self._full_numel:
                raise ValueError(
                    f"FlatParam grad numel: {self.flat_param.grad.numel()}, "
                    f"Full numel: {self._full_numel}."
                )
            elif self.flat_param.grad.numel() == self._full_numel:
                return False
        if (
            self._sharding_strategy not in GRAD_SHARD_HANDLE_STRATEGIES
            and self._dst_numel_in_cuda_for_partial_grads == self.flat_param.numel()
        ):
            logger.error(f"Check when it is needed and add the comment here.")
            return False
        return True

    def _get_padded_unsharded_flat_param(self) -> torch.Tensor:
        """Overrided to directly return the _full_param_padded"""
        return self.flat_param._full_param_padded

    def alloc_full_weights(self):
        return self._alloc_padded_unsharded_flat_param()

    def alloc_full_grads(self):
        return self._alloc_unsharded_grad()

    def _alloc_padded_unsharded_flat_param(self):
        """
        Overrided to use the buffer and to support swapping.

        Allocates the *padded* unsharded flat parameter. The unpadded unsharded
        flat parameter is always a view into the padded one. This padded
        parameter is saved to a different attribute on the ``FlatParameter``
        depending on if we force full precision.
        """
        # If flat_param is not sharded, then we don't need to allocate the
        # extra padded unsharded flat param.
        if not self.needs_full_weights_allocation():
            return
        flat_param = self.flat_param
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        if self._dst_numel_in_cuda_for_partial_weights == self._full_numel:
            # If there is no sharding, then directly load the partial weight buffer
            # which is actually the full weight buffer.
            buffer: MemoryBuffer = self._partial_weight_buffer
            assert buffer is not None, "buffer is None"
            assert buffer.numel == self._full_numel
            buffer_tensor = buffer.data
            unsharded_flat_param.data = buffer_tensor.view(unsharded_flat_param.shape)
        else:
            # Otherwise load the full weight buffer from the buffer pool.
            if self._curr_occupied_full_weights_buffer is not None:
                return unsharded_flat_param
            ring_buffer: RingMemoryBuffer = self._full_weights_ring_buffer
            buffer: MemoryBuffer = ring_buffer.get_next_buffer()
            buffer_tensor = buffer.new(self._full_numel)
            unsharded_flat_param.data = buffer_tensor.view(unsharded_flat_param.shape)
            self._curr_occupied_full_weights_buffer = buffer
            logger.debug(
                f"Allocate full weights for {self._module.name}. "
                f"Buffer name: {buffer.name}, "
                f"FlatParam numel: {self.flat_param.numel()}, "
                f"buffer numel: {buffer.numel}."
            )
        return unsharded_flat_param

    def _alloc_unsharded_grad(self):
        if not self.flat_param.requires_grad:
            return
        saved_grad_shard = getattr(self.flat_param, "_saved_grad_shard", None)
        if (
            isinstance(saved_grad_shard, torch.Tensor)
            and saved_grad_shard.numel() == self._full_numel
        ):
            self.flat_param.grad = saved_grad_shard
            return saved_grad_shard
        flat_param = self.flat_param
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        if not self._use_memory_buffer:
            raise NotImplementedError("Not implemented yet.")
        elif self._dst_numel_in_cuda_for_partial_grads == self._full_numel:
            buffer: MemoryBuffer = self._partial_grad_buffer
            assert buffer is not None, "buffer is None"
            assert buffer.numel == self._full_numel
            buffer_tensor = buffer.data
            flat_param._saved_grad_shard = buffer_tensor.view(
                unsharded_flat_param.shape
            )
        else:
            if self._curr_occupied_full_grads_buffer is not None:
                return flat_param.grad
            ring_buffer: RingMemoryBuffer = self._full_grads_ring_buffer
            buffer: MemoryBuffer = ring_buffer.get_next_buffer()
            buffer_tensor = buffer.new(self._full_numel)
            buffer_tensor.zero_()
            # Because torch doesn't support directly assigning a tensor to the grad
            # if the shapes don't match. So we use this trick to assign the grad.
            ori_flat_param_data = flat_param.data
            flat_param.data = buffer_tensor.view(unsharded_flat_param.shape)
            flat_param.grad = buffer_tensor.view(unsharded_flat_param.shape)
            flat_param.data = ori_flat_param_data
            self._curr_occupied_full_grads_buffer = buffer
            logger.debug(
                f"Allocate full grads for {self._module.name}. "
                f"Buffer name: {buffer.name}, "
                f"FlatParam numel: {self.flat_param.numel()}, "
                f"buffer numel: {buffer.numel}."
            )
        return flat_param.grad

    def swap_in_weights(self) -> bool:
        """Swap in the weights (_local_shard) to the full weight buffer."""
        # If already swapped in, then return
        swap_handle = getattr(self.flat_param._local_shard, "_swap_handle", None)
        if swap_handle is None:
            return False
        if swap_handle.numel_in_cuda() >= self._sharded_weight_numel:
            return False
        flat_param = self.flat_param
        buffer_tensor = None
        if self._use_memory_buffer:
            # Get the unsharded flat param which is where the buffer is
            unsharded_flat_param = self._get_padded_unsharded_flat_param()
            # Get the correct shard from the unsharded flat param
            buffer_tensor = unsharded_flat_param.data[: self._sharded_weight_numel]
            flat_param.data = buffer_tensor
            flat_param._local_shard.data = buffer_tensor
        # Swap in the weights
        worked = swap_(
            flat_param._local_shard,
            state="cuda",
            cache_cpu_data=True,
            cuda_buffer=buffer_tensor,
        )
        return worked

    def swap_in_grads(self) -> bool:
        """Swap in the grads (_saved_grad_shard) to the full grad buffer."""
        if not self.flat_param.requires_grad:
            return False
        if getattr(self.flat_param, "_saved_grad_shard", None) is None:
            return False
        swap_handle = getattr(self.flat_param._saved_grad_shard, "_swap_handle", None)
        if swap_handle is None:
            return False
        if swap_handle.numel_in_cuda() >= self._sharded_grad_numel:
            return False
        flat_param = self.flat_param
        buffer_tensor = None
        if self._use_memory_buffer:
            # Get the correct shard from the unsharded flat param
            if self.uses_grad_sharding_strategy:
                buffer_tensor, numel_to_pad = FlatParamHandle._get_unpadded_shard(
                    flat_param.grad, self.rank, self.world_size
                )
                assert numel_to_pad == 0, f"numel_to_pad: {numel_to_pad}"
            else:
                buffer_tensor = flat_param.grad
            flat_param._saved_grad_shard.data = buffer_tensor

        # Swap in the grads
        worked = swap_(
            flat_param._saved_grad_shard,
            state="cuda",
            cache_cpu_data=False,
            cuda_buffer=buffer_tensor,
        )
        return worked

    def swap_out_weights(self, dst_numel_in_cuda: int = None) -> bool:
        """Swap out the weights (_local_shard)"""
        flat_param = self.flat_param
        if dst_numel_in_cuda is None:
            assert self._dst_numel_in_cuda_for_partial_weights is not None, (
                "dst_numel_in_cuda is None and "
                "self._dst_numel_in_cuda_for_partial_weights is None"
            )
            dst_numel_in_cuda = self._dst_numel_in_cuda_for_partial_weights
        if dst_numel_in_cuda == flat_param.numel():
            return False
        # Get the buffer (partial weights)
        buffer_tensor = None
        if self._use_memory_buffer and self._partial_weight_buffer is not None:
            self._partial_weight_buffer.reset()
            buffer_tensor = self._partial_weight_buffer.new(dst_numel_in_cuda)
        # Swap out the weights
        worked = swap_(
            flat_param._local_shard,
            state="partial",
            dst_numel_in_cuda_for_partial=dst_numel_in_cuda,
            cache_cpu_data=False,
            cuda_buffer=buffer_tensor,
        )
        flat_param.data = flat_param._local_shard
        return worked

    def swap_out_grads(self, dst_numel_in_cuda: int = None) -> bool:
        """Swap out the grads (_saved_grad_shard)."""
        if not self.flat_param.requires_grad:
            return False
        if getattr(self.flat_param, "_saved_grad_shard", None) is None:
            return False
        flat_param = self.flat_param
        if dst_numel_in_cuda is None:
            assert self._dst_numel_in_cuda_for_partial_grads is not None, (
                "dst_numel_in_cuda is None and "
                "self._dst_numel_in_cuda_for_partial_grads is None"
            )
            dst_numel_in_cuda = self._dst_numel_in_cuda_for_partial_grads
        if dst_numel_in_cuda >= flat_param._saved_grad_shard.numel():
            return False
        # Get the buffer (partial grads)
        buffer_tensor = None
        if self._use_memory_buffer and self._partial_grad_buffer is not None:
            self._partial_grad_buffer.reset()
            buffer_tensor = self._partial_grad_buffer.new(dst_numel_in_cuda)
        # Swap out the grads
        worked = swap_(
            flat_param._saved_grad_shard,
            state="partial",
            dst_numel_in_cuda_for_partial=dst_numel_in_cuda,
            cache_cpu_data=True,
            cuda_buffer=buffer_tensor,
        )
        return worked

    ############
    # Sharding #
    ############
    @torch.no_grad()
    def shard(self):
        """
        Shards the handle's ``FlatParameter``. This allocates new memory for
        the sharded flat parameter and frees the unsharded flat parameter's
        storage.

        Postcondition: ``self.flat_param`` is the sharded flat parameter. Shard
        metadata attributes are set for all sharding strategies.
        """
        flat_param = self.flat_param
        if not self.uses_sharded_strategy:
            self._init_shard_metadata(0, 0, flat_param.numel() - 1)
        else:
            _p_assert(
                flat_param.storage_offset() == 0,
                "The `FlatParameter` is not the sole occupant of its storage",
            )
            orig_storage = flat_param._typed_storage()
            sharded_flat_param, numel_padded = FlatParamHandle._get_shard(
                flat_param, self.rank, self.world_size
            )
            flat_param.set_(sharded_flat_param)  # type: ignore[call-overload]
            start_idx = sharded_flat_param.numel() * self.rank
            end_idx = sharded_flat_param.numel() * (self.rank + 1) - 1  # inclusive
            self._init_shard_metadata(numel_padded, start_idx, end_idx)
            if orig_storage._size() > 0:
                orig_storage._resize_(0)
        if self._use_orig_params:
            self._use_sharded_views()

    def unshard(self):
        """
        Runs the unshard logic. This includes all-gathering the flat parameter
        and switching to using the unsharded flat parameter. If the handle does
        not need unsharding, then this only switches to using the unsharded
        flat parameter. For ``NO_SHARD``, this is a no-op.

        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
        if not self.needs_unshard():
            # Even when not needing an unshard, we should switch to using
            # the unsharded flat parameter
            unsharded_flat_param = (
                self._get_padded_unsharded_flat_param()
                if self.uses_sharded_strategy
                else self.flat_param
            )
            self._use_unsharded_flat_param(unsharded_flat_param)
            return
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
        self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def needs_unshard(self) -> bool:
        """Returns if the handle's flat parameter needs to be unsharded."""
        if not self.uses_sharded_strategy:
            return False
        # logger.debug(
        #     f"[{self._module.name}] FlatParam Numel: {self.flat_param.numel()}, Full Numel: {self._full_numel}"
        # )
        if self.flat_param.numel() == self._full_numel:
            return False
        return True
        # unsharded_flat_param = self._get_padded_unsharded_flat_param()
        # already_unsharded = (
        #     unsharded_flat_param._typed_storage()._size()
        #     == unsharded_flat_param.numel()
        # )
        # return not already_unsharded

    def reshard(self, free_unsharded_flat_param: bool = True):
        """
        When flat_param is resharded,
        1. the flat_param will be freed
        2. params will be empty
        3. only the local shard will be kept.
        """
        assert free_unsharded_flat_param is True
        if not self.uses_weight_sharding_strategy:
            return

        flat_param = self.flat_param
        # Point the flat_param to the local shard
        flat_param.data = flat_param._local_shard

    @property
    def _full_weights_ring_buffer(self) -> RingMemoryBuffer:
        return get_ring_memory_buffer(
            f"full_weights_dtype_{self.flat_param.dtype}_requires_grad_{self.flat_param.requires_grad}"
        )

    @property
    def _full_grads_ring_buffer(self) -> RingMemoryBuffer:
        return get_ring_memory_buffer(f"full_grads_dtype_{self.flat_param.dtype}")

    def dealloc_full_weights(self):
        return self._free_unsharded_flat_param()

    def dealloc_full_grads(self):
        if self._curr_occupied_full_grads_buffer is not None:
            logger.debug(
                f"Deallocate full grads for {self._module.name}. "
                f"Buffer name: {self._curr_occupied_full_grads_buffer.name}, "
                f"FlatParam numel: {self.flat_param.numel()}, "
                # f"FlatParam grad numel: {self.flat_param.grad.numel()}. "
            )
            self._curr_occupied_full_grads_buffer.reset()
            self._curr_occupied_full_grads_buffer = None

    def _free_unsharded_flat_param(self):
        if not self._use_memory_buffer:
            return super()._free_unsharded_flat_param()
        if self._curr_occupied_full_weights_buffer is not None:
            self._curr_occupied_full_weights_buffer.reset()
            self._curr_occupied_full_weights_buffer = None
            logger.debug(
                f"Deallocate full weights for {self._module.name}. "
                f"FlatParam numel: {self.flat_param.numel()}, "
            )

    def reduce_grad(self, stream: torch.cuda.Stream, skip=False):
        """The arg `stream` is used for the `no_dispatch_record_stream`."""
        if not self.flat_param.requires_grad:
            return
        flat_param = self.flat_param
        unsharded_grad = flat_param.grad.data
        if skip:
            new_sharded_grad = unsharded_grad
        elif not self.uses_sharded_strategy:
            # If not using sharded strategy and the grad is needed to be reduced,
            # we directly call all_reduce.
            # FIXME(zhanda): fix the bug of the sync of the grad brought by the
            # post div factor.
            # _div_if_needed(unsharded_grad, self._gradient_predivide_factor)
            dist.all_reduce(unsharded_grad, group=self.reduce_scatter_process_group)
            # _div_if_needed(unsharded_grad, self._gradient_postdivide_factor)
            new_sharded_grad = unsharded_grad
        else:
            # If using sharded strategy, we need to call reduce_scatter.
            padded_unsharded_grad, new_sharded_grad = self._get_reduce_scatter_tensors(
                unsharded_grad
            )
            # FIXME(zhanda): fix the bug of the sync of the grad brought by the
            # post div factor.
            # _div_if_needed(padded_unsharded_grad, self._gradient_predivide_factor)
            dist.reduce_scatter_tensor(
                new_sharded_grad,
                padded_unsharded_grad,
                group=self.reduce_scatter_process_group,
            )
            # _div_if_needed(new_sharded_grad, self._gradient_postdivide_factor)
        # Accumulate and post process
        # Instead of accumulating the grad. Since we have already accumulated the grad
        # because the swapped-in grads are in flat_param.grad. So we just need to
        # assign the sharded_grad to the _saved_grad_shard.
        # self._accumulate_sharded_grad(new_sharded_grad)
        if getattr(flat_param, "_saved_grad_shard", None) is None:
            flat_param._saved_grad_shard = new_sharded_grad
        else:
            flat_param._saved_grad_shard.data = new_sharded_grad
        self._post_reduce_grad_callback()
        # Since the unsharded gradient is produced in the computation
        # stream and consumed in the post-backward stream, inform the
        # caching allocator (before it goes out of scope)
        _no_dispatch_record_stream(unsharded_grad, stream)

    def _get_reduce_scatter_tensors(
        self, unsharded_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the input and output tensors to reduce-scatter, respectively.
        """
        world_size = self.world_size
        chunks = list(unsharded_grad.chunk(world_size))
        numel_to_pad = world_size * chunks[0].numel() - unsharded_grad.numel()
        padded_unsharded_grad = (
            F.pad(unsharded_grad, [0, numel_to_pad])
            if numel_to_pad > 0
            else unsharded_grad
        )
        # =============================================================
        # Original implementation
        # new_sharded_grad = torch.empty_like(chunks[0])  # padded
        new_sharded_grad = chunks[0].clone().detach()
        # =============================================================
        return padded_unsharded_grad, new_sharded_grad

    def _accumulate_sharded_grad(self, sharded_grad: torch.Tensor):
        flat_param = self.flat_param
        accumulate_grad = getattr(flat_param, "_saved_grad_shard", None) is not None
        if accumulate_grad:
            _check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard)
            flat_param._saved_grad_shard += sharded_grad
        else:
            flat_param._saved_grad_shard = sharded_grad

    def _post_reduce_grad_callback(self):
        flat_param = self.flat_param
        # assert flat_param.grad is None
        # assert flat_param._temp_flat_grad is None
        assert flat_param._saved_grad_shard is not None
        flat_param.grad = None
        # for param in flat_param._params:
        #     assert param.grad is None

    @property
    def uses_sharded_strategy(self) -> bool:
        return self._sharding_strategy != HandleShardingStrategy.NO_SHARD

    @property
    def uses_weight_sharding_strategy(self) -> bool:
        return self._sharding_strategy in [
            HandleShardingStrategy.FULL_SHARD,
        ]

    @property
    def uses_grad_sharding_strategy(self) -> bool:
        return self._sharding_strategy in [
            HandleShardingStrategy.FULL_SHARD,
            HandleShardingStrategy.OPT_AND_GRAD,
        ]

    def __repr__(self):
        return f"FlatParamHandle({self._module.name})"


def _get_aligned_numel(unsharded_dtype: torch.dtype):
    # NOTE: This alignment constraint comes from TorchInductor.
    ALIGNMENT = 16  # bytes
    unsharded_dtype_size = _get_dtype_size(unsharded_dtype)
    aligned_numel = ALIGNMENT // unsharded_dtype_size
    return aligned_numel


@functools.lru_cache(8)
def _get_dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()
