from __future__ import annotations
from contextlib import nullcontext
from enum import Enum, auto
from functools import wraps, partial
from itertools import chain, product
from types import MethodType
from typing import List, Tuple, Dict, Callable, Union, Any, Optional, Sequence, Iterable
import inspect
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils._pytree import tree_flatten
from torch.distributed.fsdp._common_utils import _no_dispatch_record_stream

from mist import parallel_state
from mist.logger import get_logger, update_logger_formatter_for_rank
from mist.re_swap_manager.flat_param import (
    DTYPE_RGRAD,
    FlatParamHandle,
    HandleShardingStrategy,
    HandleTrainingState,
    GRAD_SHARD_HANDLE_STRATEGIES,
)
from mist.re_swap_manager.mem_buffer_pool import (
    RingMemoryBuffer,
    allocate_memory_buffer,
    allocate_ring_memory_buffer,
    MemoryBuffer,
)
from mist.re_swap_manager.optimizer import ModuleOptimizerHandle, ReSwapAdamW
from mist.re_swap_manager.swap_backend import (
    swap_,
    preprocess_tensors_to_be_swapped,
    get_swapped,
    is_tensor_on_expected_device,
    DEFAULT_MEMORY_LIMIT,
    collect_stats_for_maybe_swapped_tensor
)
from mist.utils.common import torch_dtype_to_str
from mist.utils.device import get_device, stream_synchronize
from mist.utils.gradient_checkpointing import wrap_forward_with_gradient_checkpointing
from mist.utils.inspect import map_args_kwargs_to_args
from mist.utils.memory import cuda_empty_cache

logger = get_logger()

MEMORY_LIMIT = 1 * 1024 * 1024

def _debug_grads(msg, module):
    return 
    sum_grads_in_module = sum(
        p.grad.float().sum().item()
        for p in module.parameters()
        if p.grad is not None
    )
    logger.error(f"[{module.name}] {msg} - Sum of Grads: {sum_grads_in_module}")

def _possible_identifier(tensor):
    ret = []
    if tensor is None:
        return ret
    if getattr(tensor, "name", None) is not None:
        ret.append(tensor.name)
    ret.append(id(tensor))
    ret.append(tensor.data_ptr())
    if getattr(tensor, "_base", None) is not None:
        ret.append(tensor._base.data_ptr())
        ret.append(id(tensor._base))
    return ret


def _any_in(a, b):
    return any(i in b for i in a)


def _aligned_size(size, align):
    ret = (size + align - 1) // align * align
    ret = int(ret)
    return ret


class NextStepType(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class ModuleReSwapManager:
    def __init__(
        self,
        module: nn.Module,
        parent_model_manager: ModelReSwapManager,
        state_swap_ratios: Tuple[float, float],
        activation_swap_ratio: float,
        sharding_strategy: HandleShardingStrategy,
        process_group: dist.ProcessGroup,
        all_gather_process_group: Optional[dist.ProcessGroup] = None,
        reduce_scatter_process_group: Optional[dist.ProcessGroup] = None,
        activation_checkpointing: bool = True,
        cuda_device: Optional[Union[torch.device, int]] = None,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
        cpu_accumu_grad: bool = False,
        cpu_optim_step: bool = False,
        opt_swap_ratios: Optional[Dict[str, float]] = None,
        use_memory_buffer: bool = True,
    ):
        self.module = module
        self.parent_model_manager = parent_model_manager
        self.state_swap_ratios = state_swap_ratios
        self.activation_swap_ratio = activation_swap_ratio
        self.sharding_strategy = sharding_strategy
        self.process_group = process_group
        self.all_gather_process_group = all_gather_process_group or process_group
        self.reduce_scatter_process_group = (
            reduce_scatter_process_group or process_group
        )
        self.activation_checkpointing = activation_checkpointing
        self.opt_swap_ratios = opt_swap_ratios or {}
        self.cuda_device = get_device(cuda_device)
        self.grad_scaler = grad_scaler
        self.cpu_accumu_grad = cpu_accumu_grad
        self.cpu_optim_step = cpu_optim_step
        self.optimizer_handle = None

        # Record the original forward method since it will be wrapped.
        self._orig_forward = module.forward

        # Micro Batch Counter
        self.finished_forward_steps = 0
        self.finished_backward_steps = 0

        # Init flat param handles for different dtypes and requires_grad.
        # Params are saved in the form of the flat param.
        self._dtype_to_handles_with_grad: Dict[torch.dtype, FlatParamHandle] = {}
        self._dtype_to_handles_without_grad: Dict[torch.dtype, FlatParamHandle] = {}
        self._handles_with_grad: List[FlatParamHandle] = []
        self._handles_without_grad: List[FlatParamHandle] = []
        self._handles: List[FlatParamHandle] = []
        # Call the init function.
        prev_alloc = torch.cuda.memory_allocated() // 1024**2
        self._init_flat_param_handles()
        logger.debug(
            f"Init flat param handles for module {module.name}. "
            f"Prev Alloc: {prev_alloc:.2f} MB, "
            f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB."
        )

        # Init activation swapping.
        self.activation_swapping_enabled = self.activation_swap_ratio > 0.0
        if self.activation_swapping_enabled:
            self._init_activation_swapping()

        # Wrap the forward with different optimizations.
        # Outermost
        #  - Activation Swapping
        #  |- Activation Checkpointing
        #  |-- Parameter Reassignment
        # Innermost
        self._wrap_forward()

    def _verify(self):
        if self.cpu_accumu_grad:
            raise ValueError("CPU accumulation of gradients is not supported.")
        if self.cpu_optim_step:
            raise ValueError("CPU optimization step is not supported.")

    # ###############
    # Init and Wrap #
    # ###############
    def _init_flat_param_handles(self, params: Optional[Sequence[torch.Tensor]] = None):
        if params is None:
            params = list(set(p for p in self.module.parameters() if p is not None))
        dtypes_choices = set(p.dtype for p in params)
        self._dtype_to_handles_with_grad = {}
        self._dtype_to_handles_without_grad = {}
        requires_grad_choices = {
            True: self._dtype_to_handles_with_grad,
            False: self._dtype_to_handles_without_grad,
        }

        for requires_grad, dtype_to_flat_params in requires_grad_choices.items():
            for dtype in dtypes_choices:
                params = [
                    p
                    for p in params
                    if p is not None
                    and p.dtype == dtype
                    and p.requires_grad == requires_grad
                ]
                if len(params) == 0:
                    continue
                weight_swap_ratio, grad_swap_ratio = self.state_swap_ratios
                flat_param_handle = FlatParamHandle(
                    params=params,
                    module=self.module,
                    device=self.cuda_device,
                    sharding_strategy=self.sharding_strategy,
                    process_group=self.process_group,
                    all_gather_process_group=self.all_gather_process_group,
                    reduce_scatter_process_group=self.reduce_scatter_process_group,
                    param_swap_ratio=weight_swap_ratio,
                    grad_swap_ratio=grad_swap_ratio,
                )
                dtype_to_flat_params[dtype] = flat_param_handle
                self.module.register_parameter(
                    f"dtype_{torch_dtype_to_str(dtype)}_requires_grad_{requires_grad}",
                    flat_param_handle.flat_param,
                )

        self._handles_with_grad = list(self._dtype_to_handles_with_grad.values())
        self._handles_without_grad = list(self._dtype_to_handles_without_grad.values())
        self._handles = self._handles_with_grad + self._handles_without_grad

    def _wrap_forward(self):
        # Wrap the forward with different optimizations.
        # Outermost
        #  - Activation Swapping
        #  |- Activation Checkpointing
        #  |-- Parameter Reassignment
        # Innermost
        self._wrap_forward_with_param_reassignment()

        if self.activation_checkpointing:
            wrap_forward_with_gradient_checkpointing(self.module)

        if self.activation_swapping_enabled:
            self._attach_activation_swapping()

    def _wrap_forward_with_param_reassignment(self):
        """
        Note: this wrapping is not the best way. But we have to do this because we
        want to support multiple wrapping.
        """
        orig_forward_method = self.module.forward
        orig_forward_func = type(self.module).forward
        handles = self.handles()

        @wraps(orig_forward_func)
        def wrapped_forward(self, *args, **kwargs):
            # logger.debug(f"Running into param reassignment wrapper.")
            del self  # unused. for the MethodType.
            for handle in handles:
                handle._use_unsharded_views(as_params=False)
            return orig_forward_method(*args, **kwargs)

        self.module.forward = MethodType(wrapped_forward, self.module)

    # ###########
    # Utilities #
    # ###########
    def handles(self, requires_grad: Optional[bool] = None):
        if requires_grad is None:
            return self._handles
        elif requires_grad:
            return self._handles_with_grad
        else:
            return self._handles_without_grad

    def get_handle(self, dtype: torch.dtype, requires_grad: bool):
        if requires_grad:
            return self._dtype_to_handles_with_grad.get(dtype, None)
        else:
            return self._dtype_to_handles_without_grad.get(dtype, None)

    def set_training_state(self, training_state: HandleTrainingState):
        for handle in self.handles():
            handle.set_training_state(training_state)

    @property
    def total_gradient_accumulation_steps(self):
        return self.parent_model_manager.total_gradient_accumulation_steps

    @property
    def is_first_forward_micro_batch(self):
        return self.finished_forward_steps == 0

    @property
    def is_last_backward_micro_batch(self):
        return (
            self.finished_backward_steps == self.total_gradient_accumulation_steps - 1
        )

    @property
    def num_pipeline_stages(self):
        return self.parent_model_manager.num_pipeline_stages

    @property
    def pipeline_stage_idx(self):
        return self.parent_model_manager.pipeline_stage_idx

    @property
    def warmup_steps_for_pipeline_parallel(self):
        return self.num_pipeline_stages - self.pipeline_stage_idx - 1

    @staticmethod
    def toggled_step_type(
        finished_forward_steps: int, finished_backward_steps: int
    ) -> NextStepType:
        """Get the next step type given the finished forward and backward steps."""
        if finished_forward_steps < finished_backward_steps:
            raise ValueError(
                f"finished_forward_steps: {finished_forward_steps} < finished_backward_steps: {finished_backward_steps}"
            )
        if finished_forward_steps == finished_backward_steps:
            # Means the current step is the forward step.
            return NextStepType.BACKWARD
        else:  # finished_forward_steps > finished_backward_steps
            # Means the current step is the backward step.
            return NextStepType.FORWARD

    @property
    def next_step_type(self):
        """Return next step type."""
        if (
            self.num_pipeline_stages == 1
            or self.pipeline_stage_idx == self.num_pipeline_stages - 1
        ):
            # FWD -> BWD -> FWD -> BWD -> ...
            # E.g. (0, 0): next_step_type = BACKWARD
            # E.g. (1, 0): next_step_type = FORWARD
            # E.g. (1, 1): next_step_type = BACKWARD
            return self.toggled_step_type(
                self.finished_forward_steps, self.finished_backward_steps
            )
        else:
            # In pipeline parallel (except the last stage)
            # [0, warmup): next_step_type = FORWARD
            # [warmup, total_gradient_accumulation_steps * 2 - warmup): next_step_type = TOGGLED
            # [total_gradient_accumulation_steps * 2 - warmup, total_gradient_accumulation_steps * 2 - 1): next_step_type = BACKWARD
            # Last micro batch: next_step_type = FORWARD
            warmup = self.warmup_steps_for_pipeline_parallel
            total = self.total_gradient_accumulation_steps * 2
            finished_step = self.finished_forward_steps + self.finished_backward_steps
            if finished_step < warmup:
                return NextStepType.FORWARD
            elif finished_step < total - warmup - 1:
                return self.toggled_step_type(
                    self.finished_forward_steps - warmup, self.finished_backward_steps
                )
            elif finished_step < total - 1:
                return NextStepType.BACKWARD
            else:  # finished_step == total - 1
                return NextStepType.FORWARD

    def step_forward(self):
        self.finished_forward_steps += 1
        self._update_steps_info()

    def step_backward(self):
        self.finished_backward_steps += 1
        self._update_steps_info()

    def _update_steps_info(self):
        assert (
            self.finished_backward_steps
            <= self.finished_forward_steps
            <= self.total_gradient_accumulation_steps
        ), (
            f"Finished backward steps: {self.finished_backward_steps}, "
            f"Finished forward steps: {self.finished_forward_steps}, "
            f"Total gradient accumulation steps: {self.total_gradient_accumulation_steps}"
        )
        if (
            self.finished_backward_steps
            == self.finished_forward_steps
            == self.total_gradient_accumulation_steps
        ):
            self.optimizer_handle.ready_to_step = True
            self.finished_backward_steps = 0
            self.finished_forward_steps = 0

            # Check all the saved activations are released.
            if self.activation_swapping_enabled:
                assert all(
                    len(tensors) == 0 for tensors in self._saved_raw_tensors
                ), f"Saved tensors: {self._saved_raw_tensors}"
                assert all(
                    len(tensors) == 0 for tensors in self._saved_proxy_tensors
                ), f"Saved tensors: {self._saved_proxy_tensors}"

            # ################
            # Debugging Info #
            # ################
            # if hasattr(self, "_saved_tensors_debug_sum_real"):
            #     logger.error(
            #         f"[{self.module.name}] \n"
            #         f"Real Sum: {self._saved_tensors_debug_sum_real}, \n"
            #         f"Expected Sum: {self._saved_tensors_debug_sum_expected}"
            #     )
            #     del self._saved_tensors_debug_sum_real
            #     del self._saved_tensors_debug_sum_expected
            # if hasattr(self, "_saved_tensors_debug_sum_real_cpu"):
            #     logger.error(
            #         f"[{self.module.name}] \n"
            #         f"Real Sum CPU: {self._saved_tensors_debug_sum_real_cpu}"
            #     )
            #     del self._saved_tensors_debug_sum_real_cpu

    # #######################
    # Health and Allocation #
    # #######################

    def is_weight_full(self):
        is_full = True
        for handle in self.handles():
            flat_param = handle.flat_param
            is_full &= (
                flat_param.numel() == flat_param._unsharded_numel
                and flat_param._typed_storage()._size() != 0
            )
        return is_full

    def is_grad_full(self, is_sharded: bool = False):
        """
        The grad can be full but still sharded. This may sound weird but it's possible
        because for ZeRO-1, the grad shouldn't be sharded but in the last micro-batch,
        since the collective communication is reduce-scatter, the grad would be sharded./
        """
        is_full = True
        for handle in self.handles():
            flat_param = handle.flat_param
            curr_expected_numel = (
                flat_param._unsharded_numel
                if not is_sharded
                else flat_param._sharded_numel
            )
            is_full &= (
                flat_param.grad is not None
                and flat_param.grad.numel() == curr_expected_numel
                and flat_param.grad._typed_storage()._size() != 0
            )
        return is_full

    def alloc_full_weights(self):
        """Allocate the full weights for the flat param."""
        for handle in self.handles():
            handle.alloc_full_weights()

    def alloc_full_grads(self):
        """Allocate the full grads for the flat param"""
        for handle in self.handles(requires_grad=True):
            handle.alloc_full_grads()

    def dealloc_full_weights(self):
        """Deallocate the full weights for the flat param."""
        for handle in self.handles():
            handle.dealloc_full_weights()

    def dealloc_full_grads(self):
        """Deallocate the full grads for the flat param"""
        for handle in self.handles(requires_grad=True):
            handle.dealloc_full_grads()

    def zero_grad(self):
        for handle in self.handles(requires_grad=True):
            handle.flat_param.grad = None
            # logger.info(
            #     f"handle.flat_param.grad has _swap_handle: {hasattr(handle.flat_param, '_swap_handle')}"
            # )
            handle.flat_param._saved_grad = None
            # logger.info(
            #     f"handle.flat_param._saved_grad has _swap_handle: {hasattr(handle.flat_param._saved_grad, '_swap_handle')}"
            # )
            assert handle._curr_occupied_full_grads_buffer is None, (
                f"handle {handle} has not released the full grads buffer "
                f"after the last backward."
            )
            if handle._partial_grad_buffer is not None:
                handle._partial_grad_buffer.data.zero_()

    # ################
    # State Swapping #
    # ################
    def swap_in_weights(self, stream: torch.Stream):
        with torch.cuda.stream(stream):
            with torch.profiler.record_function(f"{self.module.name}.swap_in_weights"):
                worked = False
                for handle in self.handles():
                    worked |= handle.swap_in_weights(stream=stream)
        # torch.cuda.synchronize()

    def swap_out_weights(self, stream: torch.Stream):
        with torch.cuda.stream(stream):
            with torch.profiler.record_function(f"{self.module.name}.swap_out_weights"):
                worked = False
                for handle in self.handles():
                    worked |= handle.swap_out_weights(stream=stream)
        # # TODO(tmp)
        # torch.cuda.synchronize()

    def swap_in_grads(self, stream: torch.Stream, shard: bool = False):
        prev_alloc = torch.cuda.memory_allocated() // 1024**2

        with torch.profiler.record_function(f"{self.module.name}.swap_in_grads"):
            worked = False
            for handle in self.handles(requires_grad=True):
                worked |= handle.swap_in_grads(stream=stream, shard=shard)

        if worked:
            logger.debug(
                f"[{self.module.name}][FWD: {self.finished_forward_steps}, BWD: {self.finished_backward_steps}] "
                f"Swapping in grads. Prev Alloc: {prev_alloc:.2f} MB, "
                f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB."
            )

    def swap_out_grads(self, stream: torch.Stream):
        prev_alloc = torch.cuda.memory_allocated() // 1024**2

        with torch.profiler.record_function(f"{self.module.name}.swap_out_grads"):
            worked = False
            for handle in self.handles(requires_grad=True):
                worked |= handle.swap_out_grads(stream=stream)

        if worked:
            logger.debug(
                f"[{self.module.name}][FWD: {self.finished_forward_steps}, BWD: {self.finished_backward_steps}] "
                f"Swapping out grads. Prev Alloc: {prev_alloc:.2f} MB, "
                f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB."
            )

    # #####################
    # Activation Swapping #
    # #####################
    def _init_activation_swapping(self):
        self._params_and_buffers_in_layer_ptrs = set()
        for t in chain(self.module.parameters(), self.module.buffers()):
            for possible_id in _possible_identifier(t):
                self._params_and_buffers_in_layer_ptrs.add(possible_id)
        for handle in self.handles():
            possible_params = [handle.flat_param, handle.flat_param._full_param]
            for param in possible_params:
                for possible_id in _possible_identifier(param):
                    self._params_and_buffers_in_layer_ptrs.add(possible_id)

        # TODO(zhanda): support unordered backward.
        # currently only support ordered backward (which means the backward
        # would be called in the same order of the model forward).
        self._saved_raw_tensors = [
            [] for _ in range(self.total_gradient_accumulation_steps)
        ]
        self._saved_proxy_tensors = [
            [] for _ in range(self.total_gradient_accumulation_steps)
        ]

    def _attach_activation_swapping(self):
        """
        Note: this wrapping is not the best way. But we have to do this because we
        want to support multiple wrapping.
        """
        # Change the forward function to add the context manager.
        orig_forward_method = self.module.forward
        orig_forward_func = type(self.module).forward
        pack_hook = self._pack_hook
        unpack_hook = self._unpack_hook

        @wraps(orig_forward_func)
        def wrapped_forward(self, *args, **kwargs):
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                # logger.debug(f"Running into activation swapping wrapper.")
                return orig_forward_method(*args, **kwargs)

        self.module.forward = MethodType(wrapped_forward, self.module)

    def _detach_activation_swapping(self):
        self.module.forward = self._orig_forward

    def _pack_hook(self, tensor):
        if isinstance(tensor, torch.Tensor) and not _any_in(
            _possible_identifier(tensor), self._params_and_buffers_in_layer_ptrs
        ):
            curr_saved_tensors = self._saved_raw_tensors[self.finished_forward_steps]
            curr_saved_tensors.append(tensor)
        return tensor

    def _unpack_hook(self, tensor):
        return tensor

    @staticmethod
    def uniquify_tensors_by_data_ptr(
        tensors: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        memo = set()
        unique_tensors = []
        for tensor in tensors:
            data_ptr = tensor.data_ptr()
            if data_ptr in memo:
                continue
            memo.add(data_ptr)
            unique_tensors.append(tensor)
        return unique_tensors


    def alloc_activation_cpu_buffers(
        self, tensors: Sequence[torch.Tensor], swap_ratio: float
    ):
        if hasattr(self, "cpu_buffers_for_activations"):
            return
        self.cpu_buffers_for_activations = []
        for tensor in tensors:
            cpu_buffer = torch.zeros(
                tensor.numel() - int(tensor.numel() * (1 - swap_ratio)),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            self.cpu_buffers_for_activations.append(cpu_buffer)

    def swap_in_activations(self, stream: torch.Stream):
        if not self.activation_swapping_enabled:
            return

        # Whether need to swap in the activations.
        need_swap_in = False
        for tensor in self._saved_proxy_tensors[self.finished_backward_steps]:
            swap_handle = getattr(tensor, "_swap_handle", None)
            if (
                swap_handle is not None
                and swap_handle.numel_in_cuda() != tensor.numel()
            ):
                need_swap_in = True
                break

        if not need_swap_in:
            return

        with torch.profiler.record_function(f"{self.module.name}.swap_in_activations"):
            with torch.cuda.stream(stream):
                worked = False
                curr_saved_proxy_tensors = self._saved_proxy_tensors[
                    self.finished_backward_steps
                ]
                for tensor in curr_saved_proxy_tensors:
                    if tensor.numel() < MEMORY_LIMIT:
                        continue
                    worked |= swap_(
                        tensor, state="cuda", stream=stream, cache_cpu_data=False
                    )

            self._saved_raw_tensors[self.finished_backward_steps] = []
            self._saved_proxy_tensors[self.finished_backward_steps] = []

        # if worked:
        #     logger.debug(
        #         f"[{self.module.name}][FWD: {self.finished_forward_steps}, BWD: {self.finished_backward_steps}] "
        #         f"Swapping in activations, "
        #         f"Prev Alloc: {prev_alloc / 1024 ** 2:.2f} MB, "
        #         f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
        #         f"Saved Tensors: [{[len(t) for t in self._saved_raw_tensors]}], "
        #         f"Saved Proxy Tensors: [{[len(t) for t in self._saved_proxy_tensors]}]."
        #     )

        return worked

    def swap_out_activations(self, stream):
        if not self.activation_swapping_enabled:
            return

        curr_saved_proxy_tensors = preprocess_tensors_to_be_swapped(
            self._saved_raw_tensors[self.finished_forward_steps]
        )
        self._saved_proxy_tensors[self.finished_forward_steps] = (
            curr_saved_proxy_tensors
        )

        with torch.cuda.stream(stream):
            prev_alloc = torch.cuda.memory_allocated()
            with torch.profiler.record_function(
                f"{self.module.name}.swap_out_activations"
            ):
                worked = False
                curr_saved_proxy_tensors = self._saved_proxy_tensors[
                    self.finished_forward_steps
                ]
                for tensor in curr_saved_proxy_tensors:
                    if tensor.numel() < MEMORY_LIMIT:
                        continue

                    worked |= swap_(
                        tensor,
                        state="partial",
                        dst_numel_in_cuda_for_partial=int(
                            tensor.numel() * (1 - self.activation_swap_ratio)
                        ),
                        stream=stream,
                        cache_cpu_data=False,
                        # cpu_buffer=cpu_buffer,
                    )

            # fn = lambda x: x.float().abs().sum().item()
            # for i, tensor in enumerate(curr_saved_proxy_tensors):
            #     if tensor.numel() < MEMORY_LIMIT:
            #         continue
            #     logger.error(
            #         f"[Swap out - {self.module.name}] [{i}] {tensor.shape=}, {collect_stats_for_maybe_swapped_tensor(tensor, fn)=}"
            #     )

        # if worked:
        #     del curr_saved_proxy_tensors
        #     del tensor
        #     logger.debug(
        #         f"[{self.module.name}][FWD: {self.finished_forward_steps}, BWD: {self.finished_backward_steps}] "
        #         f"Swapping out activations, "
        #         f"Prev Alloc: {prev_alloc / 1024 ** 2:.2f} MB, "
        #         f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
        #         f"Saved Tensors: [{[len(t) for t in self._saved_raw_tensors]}], "
        #         f"Saved Proxy Tensors: [{[len(t) for t in self._saved_proxy_tensors]}]."
        #     )

        return worked

    # ##########
    # Sharding #
    # ##########
    def unshard(self, stream: torch.cuda.Stream):
        with torch.profiler.record_function(f"{self.module.name}.unshard"):
            for handle in self.handles():
                handle.unshard(stream=stream)

    def reshard(self):
        """Reshard the flat_params."""
        with torch.profiler.record_function(f"{self.module.name}.reshard"):
            for handle in self.handles():
                handle.reshard()

    def reduce_grad(self, stream: torch.cuda.Stream, skip: bool = False):
        with torch.profiler.record_function(
            f"{self.module.name}.post_backward_grad_reduce"
        ):
            for handle in self.handles():
                handle.reduce_grad(stream=stream, skip=skip)

    # #####################
    # Utils for Debugging #
    # #####################
    def log_flat_param_info(self, msg: str = ""):
        # Only render if the logger is in debug mode.
        if logger.level == logging.DEBUG:
            # logger.debug("")
            # header = f"========== Module [{self.module.name}] FWD: {self.finished_forward_steps}, BWD: {self.finished_backward_steps}] [{msg}] =========="
            # logger.debug(header)
            # logger.debug(
            #     f"|- Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB. Peak Alloc: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB."
            # )
            # for i, handle in enumerate(self.handles()):
            #     flat_param = handle.flat_param
            #     logger.debug(
            #         f"|- Handle {i} ({handle.sharding_strategy}, Swap Ratio: ({handle.param_swap_ratio:.4f}, {handle.grad_swap_ratio:.4f})): "
            #     )
            #     logger.debug(
            #         f"    |- Flat Param: {flat_param.shape}, Storage Size: {flat_param._typed_storage()._size()} "
            #     )
            #     logger.debug(
            #         f"    |- Full Param: {flat_param._full_param.shape}, Storage Size: {flat_param._full_param._typed_storage()._size()} "
            #     )
            #     logger.debug(
            #         f"    |- Local Shard: {flat_param._local_shard.shape}, Storage Size: {flat_param._local_shard._typed_storage()._size()} "
            #     )
            #     uses_full_buffer = handle._curr_occupied_full_weights_buffer is not None
            #     logger.debug(
            #         f"    |- Using Full Memory Buffer: {uses_full_buffer}, "
            #         f"is _full_param: {uses_full_buffer and handle._curr_occupied_full_weights_buffer.data.data_ptr() == flat_param._full_param.data_ptr()}"
            #     )
            # logger.debug("=" * len(header))
            # logger.debug("")

            self.log_memory_info(msg)

    def log_memory_info(self, msg: str = ""):
        if logger.level == logging.DEBUG:
            logger.debug(
                f"[{self.module.name}][FWD: {self.finished_forward_steps}, BWD: {self.finished_backward_steps}] "
                f"[{msg}] "
                f"Memory Info: "
                f"Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
                f"Peak Alloc: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB, "
                f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB, "
                f"Peak Reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB, "
                f"Ratio: {torch.cuda.max_memory_allocated() / torch.cuda.max_memory_reserved():.4f}"
            )

    def debug_log_flat_param_and_grad_value(self, msg: str = ""):
        handle = self.handles()[0]
        logger.error(
            f"[{self.module.name}] FWD: {self.finished_forward_steps}, BWD: {self.finished_backward_steps}] "
            f"Param: {handle.flat_param.sum().item():.6f}\t Grad: {handle.flat_param.grad.sum().item() if handle.flat_param.grad is not None else 0:.6f}"
        )

    def debug_maybe_swapped_flat_param_values(self, fn: Callable):
        # This function calls torch.cuda.empty_cache() and thus should be used with caution.
        # It's only used for debugging.
        flat_param = self.handles()[0].flat_param
        return collect_stats_for_maybe_swapped_tensor(flat_param._local_shard, fn)


class ModelReSwapManager:
    """
    ModelReSwapManager is the manager for the model that supports swapping and sharding.

    To use the ModelReSwapManager, the model should be constructed in a sequential
    way, i.e., the sequence of submodules. Each submodule would be managed by a
    ModuleReSwapManager. The ModuleReSwapManager would create flat_params for each
    submodule.
    """

    def __init__(
        self,
        model: nn.Module,
        modules: Dict[str, nn.Module],
        module_sequence: Sequence[str],
        state_swap_ratios: Dict[str, Tuple[float, float]],
        activation_swap_ratios: Dict[str, float],
        sharding_strategies: Dict[str, HandleShardingStrategy],
        process_groups: Dict[str, dist.ProcessGroup],
        all_gather_process_groups: Optional[Dict[str, dist.ProcessGroup]] = None,
        reduce_scatter_process_groups: Optional[Dict[str, dist.ProcessGroup]] = None,
        gradient_accumulation_steps: int = 1,
        pipeline_stage_idx: Optional[int] = None,
        num_pipeline_stages: Optional[int] = None,
        cuda_device: Optional[Union[torch.device, int]] = None,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
        cpu_accumu_grad: bool = False,
        cpu_optim_step: bool = False,
        opt_swap_ratios: Optional[Dict[str, float]] = None,
    ):
        self.model: nn.Module = model
        self.modules: Dict[str, nn.Module] = modules
        self.module_sequence: Sequence[str] = module_sequence
        # Build helper mappings.
        self.num_modules = len(module_sequence)
        self.module2idx = {name: idx for idx, name in enumerate(module_sequence)}
        self.idx2module = {idx: name for idx, name in enumerate(module_sequence)}
        overlapped_pairs = [
            (curr_module_name, next_module_name)
            for curr_module_name, next_module_name in zip(
                module_sequence[:-1], module_sequence[1:]
            )
        ]
        overlapped_pairs.append((None, module_sequence[0]))
        overlapped_pairs.append((module_sequence[-1], None))
        self.overlapped_pairs = overlapped_pairs
        # Strategy and ratios.
        self.state_swap_ratios: Dict[str, float] = state_swap_ratios
        self.activation_swap_ratios: Dict[str, float] = activation_swap_ratios
        self.sharding_strategies: Dict[str, HandleShardingStrategy] = (
            sharding_strategies
        )
        self.process_groups: Dict[str, dist.ProcessGroup] = process_groups
        self.all_gather_process_groups: Dict[str, dist.ProcessGroup] = (
            all_gather_process_groups or self.process_groups
        )
        self.reduce_scatter_process_groups: Dict[str, dist.ProcessGroup] = (
            reduce_scatter_process_groups or self.process_groups
        )
        self.total_gradient_accumulation_steps: int = gradient_accumulation_steps
        self.pipeline_stage_idx: Optional[int] = pipeline_stage_idx
        self.num_pipeline_stages: Optional[int] = num_pipeline_stages or 1
        self.cuda_device: torch.device = get_device(cuda_device)
        self.grad_scaler: Optional[torch.cuda.amp.GradScaler] = grad_scaler
        self.cpu_accumu_grad: bool = cpu_accumu_grad
        self.cpu_optim_step: bool = cpu_optim_step
        self.opt_swap_ratios: Dict[str, float] = opt_swap_ratios or {}
        # ========================
        # Deprecated attrs
        self.overlap = True
        self.use_memory_buffer = True
        # ========================
        self._verify()

        # Get first the last module name.
        self.first_module_name = module_sequence[0]
        self.last_module_name = module_sequence[-1]

        # Extra flags.
        # This is for the case that the first model's ModuleBackwardHook does not
        # work properly.
        self.use_post_accumulate_grad_hook: Dict[str, bool] = {}

        # Flat param handles for each module.
        self.module_managers: Dict[str, ModuleReSwapManager] = {}
        self.hooks = []

        # Hooks and streams.
        self._init_streams()

        # Optimizer
        self.model_optimizer = None
        self.module_optimizers: Dict[str, ModuleOptimizerHandle] = {}

        # Memory buffers
        self.ring_buffers_for_full_weights: Dict[DTYPE_RGRAD, RingMemoryBuffer] = {}
        self.ring_buffers_for_full_grads: Dict[DTYPE_RGRAD, RingMemoryBuffer] = {}
        self.optim_states_buffers: List[torch.Tensor] = []
        self.init_full_memory_buffers()

    # ###############
    # Init and Util #
    # ###############
    def _verify(self):
        for name, module in self.modules.items():
            assert name == module.name, (
                f"Module {module} does not have a name. "
                f"Please set the name of the module."
            )
            assert name in self.sharding_strategies, (
                f"Module {name} does not have a sharding strategy. "
                f"Please set the sharding strategy of the module."
            )
            assert name in self.state_swap_ratios, (
                f"Module {name} does not have a state swap ratio. "
                f"Please set the state swap ratio of the module."
            )
            assert name in self.activation_swap_ratios, (
                f"Module {name} does not have a activation swap ratio. "
                f"Please set the activation swap ratio of the module."
            )
            assert name in self.process_groups, (
                f"Module {name} does not have a process group. "
                f"Please set the process group of the module."
            )
            assert name in self.all_gather_process_groups, (
                f"Module {name} does not have a all gather process group. "
                f"Please set the all gather process group of the module."
            )
            assert name in self.reduce_scatter_process_groups, (
                f"Module {name} does not have a reduce scatter process group. "
                f"Please set the reduce scatter process group of the module."
            )
            assert name in self.opt_swap_ratios, (
                f"Module {name} does not have a opt swap ratio. "
                f"Please set the opt swap ratio of the module."
            )

        if self.cpu_optim_step:
            raise NotImplementedError(
                "CPU optim step is not supported in ModelReSwapManager."
            )

        if self.cpu_accumu_grad:
            raise NotImplementedError(
                "CPU accumulate grad is not supported in ModelReSwapManager."
            )

    def _init_streams(self):
        high_priority = -1
        self.compute_stream = torch.cuda.current_stream()
        if self.overlap:
            self.swap_in_stream = torch.cuda.Stream(priority=high_priority)
            self.swap_out_stream = torch.cuda.Stream(priority=high_priority)
            self.unshard_stream = torch.cuda.Stream(priority=high_priority)
            self.post_backward_reduce_scatter_stream = torch.cuda.Stream(
                priority=high_priority
            )
            self.opt_pre_and_step_stream = torch.cuda.Stream(priority=high_priority)
            self.opt_post_stream = torch.cuda.Stream(priority=high_priority)
            self.all_reduce_stream = self.post_backward_reduce_scatter_stream
        else:
            self.swap_in_stream = self.compute_stream
            self.swap_out_stream = self.compute_stream
            self.unshard_stream = self.compute_stream
            self.post_backward_reduce_scatter_stream = self.compute_stream
            self.opt_pre_and_step_stream = self.compute_stream
            self.opt_post_stream = self.compute_stream
            self.all_reduce_stream = self.compute_stream

    def init_module(
        self,
        module: nn.Module,
        activation_checkpointing: bool = False,
        optimizer_cls: Optional[Callable] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        is_first=False,
        is_last=False,
        use_post_accumulate_grad_hook=False,
    ):
        name = module.name
        prev_alloc = torch.cuda.memory_allocated() // 1024**2

        self._init_module(
            module=module,
            activation_checkpointing=activation_checkpointing,
            is_first=is_first,
            is_last=is_last,
            use_post_accumulate_grad_hook=use_post_accumulate_grad_hook,
        )
        self.module_managers[name].log_flat_param_info("After Module Init")

        assert optimizer_cls is not None
        assert optimizer_kwargs is not None
        assert module.name in self.opt_swap_ratios, (
            f"Module {module.name} does not have a opt swap ratio. "
            f"Please set the opt swap ratio of the module."
        )
        self._init_optimizer(
            module=module,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.module_managers[name].log_flat_param_info("After Optimizer Init")

        # Swapout the weights if needed after the optimizer initialization.
        weight_swap_ratio = self.state_swap_ratios[name][0]
        if weight_swap_ratio > 0.0:
            self.module_managers[name].swap_out_weights(
                stream=torch.cuda.current_stream()
            )

        torch.cuda.synchronize()

        logger.debug(
            f"[{module.name}] Init module manager and optimizer. "
            f"Prev Alloc: {prev_alloc:.2f} MB, "
            f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
            f"Delta: {torch.cuda.memory_allocated() / 1024 ** 2 - prev_alloc:.2f} MB."
        )
        self.module_managers[name].log_flat_param_info("After Init")

    def _init_module(
        self,
        module: nn.Module,
        activation_checkpointing: bool = False,
        is_first=False,
        is_last=False,
        use_post_accumulate_grad_hook=False,
    ):
        # prev_alloc = torch.cuda.memory_allocated() // 1024**2

        if not hasattr(module, "name"):
            raise ValueError(f"Module {module} does not have a name.")

        name = module.name
        assert name in self.modules and self.modules[name] is module
        state_swap_ratio = self.state_swap_ratios[name]
        weight_swap_ratio, grad_swap_ratio = state_swap_ratio
        activation_swap_ratio = self.activation_swap_ratios[name]
        sharding_strategy = self.sharding_strategies[name]
        process_group = self.process_groups[name]
        all_gather_process_group = self.all_gather_process_groups[name]
        reduce_scatter_process_group = self.reduce_scatter_process_groups[name]
        device = self.cuda_device

        # Set the first the the last module
        if is_first and self.first_module_name is not None:
            assert name == self.first_module_name, (
                f"First module name {self.first_module_name} is not equal to the "
                f"current module name {name}."
            )
        elif is_first and self.first_module_name is None:
            self.first_module_name = name
        if is_last and self.last_module_name is not None:
            assert name == self.last_module_name, (
                f"Last module name {self.last_module_name} is not equal to the "
                f"current module name {name}."
            )
        elif is_last and self.last_module_name is None:
            self.last_module_name = name

        # Set the attributes of the module.
        self.use_post_accumulate_grad_hook[name] = use_post_accumulate_grad_hook

        # Construct the flat param handles for the module.
        module_re_swap_manager = ModuleReSwapManager(
            module=module,
            parent_model_manager=self,
            state_swap_ratios=state_swap_ratio,
            activation_swap_ratio=activation_swap_ratio,
            sharding_strategy=sharding_strategy,
            process_group=process_group,
            all_gather_process_group=all_gather_process_group,
            reduce_scatter_process_group=reduce_scatter_process_group,
            activation_checkpointing=activation_checkpointing,
            cuda_device=device,
            grad_scaler=self.grad_scaler,
            cpu_accumu_grad=self.cpu_accumu_grad,
            cpu_optim_step=self.cpu_optim_step,
            use_memory_buffer=self.use_memory_buffer,
        )
        self.module_managers[name] = module_re_swap_manager

        torch.cuda.synchronize()

    def _init_optimizer(
        self,
        module: nn.Module,
        optimizer_cls: Callable,
        optimizer_kwargs: Dict[str, Any] = None,
    ):

        # prev_alloc = torch.cuda.memory_allocated() // 1024**2

        # TODO(zhanda): currently we only support params as all the params in the
        # module. We should support more general cases.
        assert module.name in self.modules, (
            f"Module {module.name} is not initialized. "
            f"Please initialize the module first."
        )
        name = module.name
        process_group = self.process_groups[name]
        sharding_strategy = self.sharding_strategies[name]
        module_manager = self.module_managers[name]
        swap_ratio = self.opt_swap_ratios[name]

        # Create the optimizer.
        module_optimizer_handle = ModuleOptimizerHandle(
            module,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            module_manager=module_manager,
            process_group=process_group,
            swap_ratio=swap_ratio,
            sharding_strategy=sharding_strategy,
            cpu_optim_step=self.cpu_optim_step,
            grad_scaler=self.grad_scaler,
        )
        self.module_optimizers[name] = module_optimizer_handle
        self.module_managers[name].optimizer_handle = module_optimizer_handle

        # if swap_ratio > 0.0:
        #     module_optimizer_handle.postprocess_swap_out(
        #         stream=torch.cuda.current_stream()
        #     )
        torch.cuda.synchronize()

        # logger.debug(
        #     f"[{module.name}] Init optimizer. "
        #     f"Prev Alloc: {prev_alloc:.2f} MB, "
        #     f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
        #     f"Delta: {torch.cuda.memory_allocated() / 1024 ** 2 - prev_alloc:.2f} MB."
        # )

    def init_full_memory_buffers(self):
        """
        Create full memory buffers. This works for the case that at least one of the
        swapping and sharding strategies is enabled. For each module, params will be
        categorized into (dtype, requires_grad) groups. This function will create two
        full memory buffers for each dtype group.
        """
        dtypes = set(p.dtype for p in self.model.parameters())
        rgrad = [True, False]
        for dtype, requires_grad in product(dtypes, rgrad):
            key = (dtype, requires_grad)
            modules_need_full_weights = []
            modules_need_full_grads = []
            full_weights_numel = 0
            full_grads_numel = 0
            for name, module in self.modules.items():
                sharding_strategy = self.sharding_strategies[name]
                state_swap_ratio = self.state_swap_ratios[name]
                weight_sharding = sharding_strategy == HandleShardingStrategy.FULL_SHARD
                grad_sharding = sharding_strategy in GRAD_SHARD_HANDLE_STRATEGIES
                managed_tensors = set(
                    p
                    for p in module.parameters()
                    if p is not None
                    and p.dtype == dtype
                    and p.requires_grad == requires_grad
                )
                numel = sum(p.numel() for p in managed_tensors)
                # For weights
                if weight_sharding or state_swap_ratio[0] != 0.0:
                    full_weights_numel = max(full_weights_numel, numel)
                    modules_need_full_weights.append(name)
                # For grads
                if requires_grad:
                    if grad_sharding or state_swap_ratio[1] != 0.0:
                        full_grads_numel = max(full_grads_numel, numel)
                        modules_need_full_grads.append(name)
            if full_weights_numel > 0:
                self.ring_buffers_for_full_weights[key] = allocate_ring_memory_buffer(
                    name=f"full_weights_dtype_{dtype}_requires_grad_{requires_grad}",
                    num_buffers=2,
                    numel=full_weights_numel,
                    dtype=dtype,
                    track_usage=True,
                )
            if full_grads_numel > 0:
                self.ring_buffers_for_full_grads[key] = allocate_ring_memory_buffer(
                    name=f"full_grads_dtype_{dtype}",
                    num_buffers=2,
                    numel=full_grads_numel,
                    dtype=dtype,
                    track_usage=True,
                )

    def _init_optim_states_buffers(self):
        if hasattr(self, "optim_states_buffers") and self.optim_states_buffers:
            return
        max_numel_fp32_params = 0
        for name, module in self.modules.items():
            module_manager = self.module_managers[name]
            optim_handle = self.module_optimizers[name]
            if optim_handle.swap_ratio == 0.0:
                continue
            assert len(optim_handle.fp32_params_list) == 1, (
                f"Module {name} has more than one fp32_params. "
                f"Currently we only support one fp32_params. "
                f"Lenght of fp32_params_list: {len(optim_handle.fp32_params_list)}."
            )
            assert len(optim_handle.opt_states_list) == 2, (
                f"Module {name} has more than two opt_states. "
                f"Currently we only support two opt_states. "
                f"Lenght of opt_states_list: {len(optim_handle.opt_states_list)}."
            )
            max_numel_fp32_params = max(
                max_numel_fp32_params, optim_handle.fp32_params_list[0].numel()
            )
        if max_numel_fp32_params > 0:
            self.optim_states_buffers = [
                torch.zeros(
                    max_numel_fp32_params, dtype=torch.float32, device=self.cuda_device
                ) for _ in range(3)
            ]
            return self.optim_states_buffers
        
        return None

    def _dealloc_optim_states_buffers(self):
        total_mem = torch.cuda.get_device_properties(0).total_memory
        curr_reserved = torch.cuda.memory_reserved()
        if curr_reserved / total_mem >= 0.95:
            logger.info(
                f"Current memory usage is {curr_reserved / total_mem:.2f}. "
                f"Empty CUDA Cache."
            )
            cuda_empty_cache()

        if not hasattr(self, "optim_states_buffers") or not self.optim_states_buffers:
            return
        for buffer in self.optim_states_buffers:
            buffer._typed_storage()._resize_(0)
        self.optim_states_buffers = []
        
        if curr_reserved / total_mem >= 0.95:
            logger.info(
                f"Current memory usage is {curr_reserved / total_mem:.2f}. "
                f"Empty CUDA Cache."
            )
            cuda_empty_cache()

    def get_optimizer(self):
        if self.model_optimizer is None:
            self.model_optimizer = ReSwapAdamW(self)
        return self.model_optimizer

    def reconstruct_full_state(
        self,
        module,
        training_state: Optional[HandleTrainingState] = None,
    ):
        if not training_state in [
            None,
            HandleTrainingState.FORWARD,
            HandleTrainingState.BACKWARD_PRE,
        ]:
            raise ValueError(
                f"training_state should be either FORWARD or BACKWARD_PRE. "
                f"Got {training_state}"
            )

        with torch.profiler.record_function(f"{module.name}.reconstruct_full_state"):
            # Reconstruction with the current stream
            stream = self.compute_stream
            module_manager: ModuleReSwapManager = self.module_managers[module.name]
            module_manager.set_training_state(training_state)
            module_manager.alloc_full_weights()
            module_manager.swap_in_weights(stream)
            module_manager.unshard(stream)
            if training_state == HandleTrainingState.BACKWARD_PRE:
                # Reconstruct the full grads
                module_manager.alloc_full_grads()
                # Reconstruct the grads
                module_manager.swap_in_grads(stream)
                # Reconstruct the activations
                worked = module_manager.swap_in_activations(stream)

    # #######
    # Hooks #
    # #######
    def register_hooks(self):
        # Register hooks for the the modules
        for curr_module_name, next_module_name in self.overlapped_pairs:
            curr_module = (
                self.modules[curr_module_name] if curr_module_name is not None else None
            )
            next_module = (
                self.modules[next_module_name] if next_module_name is not None else None
            )
            # register forward hooks
            if curr_module is not None:
                # forward_pre_hook = partial(
                #     self._forward_pre_hook, curr_module, next_module
                # )
                curr_module_idx = self.module2idx[curr_module_name]
                forward_pre_hook = partial(self._forward_pre_hook, curr_module_idx)
                forward_hook = partial(self._forward_hook, curr_module, next_module)
                self.hooks.append(
                    curr_module.register_forward_pre_hook(forward_pre_hook)
                )
                self.hooks.append(curr_module.register_forward_hook(forward_hook))
            # register backward hooks
            # NOTE(zhanda Dec/06/2023): the backward hook is tricky. (1) If we use the
            # `register_full_backward_hook`, there is no guarantee that the grads of
            # the params of the module are already accumulated. (2) We can use the
            # `register_post_accumulate_grad_hook` to guarantee that the grads of the
            # params are already accumulated. However, when it is fired is not clear,
            # and thus it does not match our expectation for the static planning.
            # Conclusion: that is to say, if the grads are not accumulated, our tool
            # will fail and thus the user should be aware of this. So we choose to us
            if next_module is not None:
                backward_pre_hook = partial(
                    self._backward_pre_hook, next_module, curr_module
                )
                # If the inputs do not require grad, module's backward hook will be immediately
                # fired before the grads of the weights are computed.
                backward_hook = partial(self._backward_hook, next_module, curr_module)
                self.hooks.append(
                    next_module.register_full_backward_pre_hook(backward_pre_hook)
                )
                if not self.use_post_accumulate_grad_hook[next_module_name]:
                    self.hooks.append(
                        next_module.register_full_backward_hook(backward_hook)
                    )
                else:
                    logger.debug(
                        f"Module {next_module.name} does not require grad. "
                        f"Heuristiclly register the backward hook to the grad of the first param."
                    )
                    # Get the first param that requires grad and register the hook to it.
                    first_param = next(
                        p for p in next_module.parameters() if p.requires_grad
                    )
                    assert (
                        first_param is not None
                    ), f"Module {next_module.name} does not have any param that requires grad."
                    self.hooks.append(
                        first_param.register_post_accumulate_grad_hook(backward_hook)
                    )

    def _forward_pre_hook(
        self,
        module_idx: int,
        module: nn.Module,
        module_inputs: Tuple[torch.Tensor],
    ):
        if module_idx < 0 or module_idx >= self.num_modules:
            raise ValueError(
                f"Module index {module_idx} is out of the range [0, {self.num_modules})."
            )
        curr_module = self.modules[self.idx2module[module_idx]]
        next_module = (
            self.modules[self.idx2module[module_idx + 1]]
            if module_idx + 1 < self.num_modules
            else None
        )
        next_next_module = (
            self.modules[self.idx2module[module_idx + 2]]
            if module_idx + 2 < self.num_modules
            else None
        )

        if module.name != curr_module.name:
            raise ValueError(
                f"Module {module.name} does not match the expected module {curr_module.name}."
            )
        del module  # unused

        curr_module_manager = self.module_managers[curr_module.name]
        curr_module_manager.log_flat_param_info("Before Forward")

        # (1) stream synchronization:
        stream_synchronize(
            self.compute_stream,
            self.swap_in_stream,
            self.unshard_stream,
            self.opt_pre_and_step_stream,
        )

        # Debugging!
        # for i, tensor in enumerate(module_inputs):
        #     if isinstance(tensor, torch.Tensor):
        #         logger.error(
        #             f" [{curr_module.name}] PreFwd. Input Tensor {i}: {tuple(tensor.shape)}, {tensor.float().sum().item():.6f}"
        #         )
        # for i, handle in enumerate(curr_module_manager.handles()):
        #     logger.error(
        #         f" [{curr_module.name}] PreFwd. Handle {i}: Param: {handle.flat_param.sum().item():.6f}"
        #     )

        # (2) Before the forward step of the first micro-batch, do the optimizer step.
        if curr_module_manager.is_first_forward_micro_batch:
            assert curr_module.name in self.module_optimizers, (
                f"Module {curr_module.name} does not have an optimizer handle. "
                f"Please initialize the optimizer handle for the module."
            )
            curr_optim_handle = self.module_optimizers[curr_module.name]
            if curr_optim_handle.ready_to_step:
                # Only the first module should step itself.
                assert (
                    curr_module.name == self.first_module_name
                ), f"Only the first module should step itself. "
                self._init_optim_states_buffers()
                cuda_buffers = self.optim_states_buffers
                # curr_module_manager.log_flat_param_info("Before curr optim step")
                curr_optim_handle.preprocess(self.compute_stream, cuda_buffers)
                # curr_module_manager.log_flat_param_info("After curr optim preprocess")
                curr_optim_handle.step()
                # curr_module_manager.log_flat_param_info("After curr optim step")
                curr_optim_handle.postprocess_copy_main_params_to_model_params(
                    self.compute_stream
                )
                # curr_module_manager.log_flat_param_info(
                #     "After curr optim copy to model"
                # )
                curr_optim_handle.postprocess_swap_out(self.compute_stream)
                # curr_module_manager.log_flat_param_info("After curr optim swap out")
                curr_optim_handle.ready_to_step = False
                if next_module is not None:
                    next_optim_handle = self.module_optimizers[next_module.name]
                    if next_optim_handle.ready_to_step:
                        next_optim_handle.preprocess(self.compute_stream, cuda_buffers)
            self.swap_in_stream.wait_stream(self.compute_stream)
            cuda_buffers = self.optim_states_buffers
            # Process the next module's optimizer handle.
            if next_module is not None:
                assert next_module.name in self.module_optimizers
                next_optim_handle = self.module_optimizers[next_module.name]
                next_module_manager = self.module_managers[next_module.name]
                if next_optim_handle.ready_to_step:
                    # # with torch.cuda.stream(self.opt_pre_and_step_stream):
                    # next_optim_handle.preprocess(self.opt_pre_and_step_stream)
                    # # next_module_manager.log_flat_param_info(
                    # #     "Before next optim step"
                    # # )
                    # self.compute_stream.wait_stream(self.opt_pre_and_step_stream)
                    # next_optim_handle.step()
                    # self.module_managers[next_module.name].alloc_full_weights()
                    # # next_module_manager.log_flat_param_info(
                    # #     "After optim alloc weights"
                    # # )
                    # self.opt_pre_and_step_stream.wait_stream(self.compute_stream)
                    # next_optim_handle.postprocess_copy_main_params_to_model_params(
                    #     self.opt_pre_and_step_stream
                    # )
                    # # next_module_manager.log_flat_param_info(
                    # #     "After optim copy to model"
                    # # )
                    # next_optim_handle.postprocess_swap_out(self.opt_pre_and_step_stream)
                    # next_module_manager.log_flat_param_info("After optim swap out")
                    # self.compute_stream.wait_stream(self.opt_pre_and_step_stream)
                    # ========================
                    with torch.cuda.stream(self.swap_in_stream):
                        # next_optim_handle._debug_compute_main_params(
                        #     f"0000 - {next_module.name}"
                        # )
                        next_optim_handle.step()
                        # next_optim_handle._debug_compute_main_params(
                        #     f"1111 - {next_module.name}"
                        # )
                        next_module_manager.alloc_full_weights()
                        # next_optim_handle._debug_compute_main_params(
                        #     f"2222 - {next_module.name}"
                        # )
                        # self.opt_pre_and_step_stream.wait_stream(self.swap_in_stream)
                        # next_optim_handle._debug_compute_main_params(
                        #     f"3333 - {next_module.name}"
                        # )
                        next_optim_handle.postprocess_copy_main_params_to_model_params(
                            self.swap_in_stream
                        )
                        next_optim_handle.postprocess_swap_out(self.swap_in_stream)
                        self.swap_in_stream.record_event()
                        next_optim_handle.ready_to_step = False
                    # ========================
                    if next_next_module is not None:
                        next_next_optim_handle = self.module_optimizers[
                            next_next_module.name
                        ]
                        if next_next_optim_handle.ready_to_step:
                            with torch.cuda.stream(self.swap_in_stream):
                                next_next_optim_handle.preprocess(self.swap_in_stream, cuda_buffers)

        # (2) check curr_module's health.
        # If not healthy, recover by swapping and unsharding.
        # This is a fallback mechanism and should not be triggered frequently.
        curr_module_manager.log_flat_param_info(f"Before Recover")
        self.reconstruct_full_state(
            curr_module,
            training_state=HandleTrainingState.FORWARD,
        )
        curr_module_manager.log_flat_param_info(f"After Recover")

        # If ALWAYS_BACKWARD, this will not be triggered.
        # If ALWAYS_FORWARD, and the next_module is None, we should forward to the
        # first module.
        # If FORWARD_BACKWARD, and being the last module, no need to swap in
        if next_module is None:
            assert curr_module.name == self.last_module_name
            if curr_module_manager.next_step_type == NextStepType.FORWARD:
                next_module = self.modules[self.first_module_name]
            elif curr_module_manager.next_step_type == NextStepType.BACKWARD:
                # This is the last module, call backward swap in to swap in the gradients.
                curr_module_manager.alloc_full_grads()
                curr_module_manager.swap_in_grads(self.swap_in_stream)
                return

        next_module_manager = self.module_managers[next_module.name]
        next_module_manager.set_training_state(HandleTrainingState.FORWARD)
        next_module_manager.log_flat_param_info("Before Next Module Alloc In Forward")
        next_module_manager.alloc_full_weights()
        next_module_manager.log_flat_param_info("After Next Module Alloc In Forward")

        # (3) launch next_module's swap_in (non-blocking) after curr_module's computation
        next_module_manager.log_flat_param_info("Before Next Module Swap In Forward")
        with torch.cuda.stream(self.swap_in_stream):
            next_module_manager.swap_in_weights(self.swap_in_stream)
        self.swap_in_stream.record_event()
        next_module_manager.log_flat_param_info("After Next Module Swap In Forward")

        # (4) launch next_module's unsharding (non-blocking) after swap_in
        next_module_manager.log_flat_param_info("Before Next Module Unshard Forward")
        self.unshard_stream.wait_stream(self.swap_in_stream)
        next_module_manager.unshard(self.swap_in_stream)
        self.unshard_stream.record_event()
        next_module_manager.log_flat_param_info("After Next Module Unshard Forward")


    def _forward_hook(
        self,
        curr_module: nn.Module,
        next_module: Optional[nn.Module],
        module,
        input,
        output,
    ):
        assert (
            module.name == curr_module.name
        ), f"Module {module.name} does not match the expected module {curr_module.name}."
        del module  # unused
        curr_module_manager = self.module_managers[curr_module.name]
        curr_module_manager.log_flat_param_info("FWD Post")

        # If it is the last module, deallocate buffers for states
        if curr_module.name == self.last_module_name and self.optim_states_buffers:
            self._dealloc_optim_states_buffers()

        # logger.debug(
        #     f"[{curr_module.name}][FWD: {curr_module_manager.finished_forward_steps}, BWD: {curr_module_manager.finished_backward_steps}] "
        #     f"Forward post hook begins. "
        #     f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
        #     f"Peak Alloc: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB."
        # )
        curr_module_manager = self.module_managers[curr_module.name]
        curr_module_manager.set_training_state(HandleTrainingState.FORWARD)
        # (1) stream synchronization:
        # wait for the prev_module's swap_out
        # curr_module's swap_out should wait for the curr_module's computation
        stream_synchronize(self.compute_stream, self.swap_out_stream)

        # (2) If the next module is backward and it is the last module, don't do anything.

        # If ALWAYS_BACKWARD, this will not be triggered.
        # If ALWAYS_FORWARD, reshard the curr_module's params
        # If FORWARD_BACKWARD and being the last module, don't reshard
        if (
            next_module is None
            and curr_module_manager.next_step_type == NextStepType.BACKWARD
        ):
            assert curr_module.name == self.last_module_name
            # Only if the saved tensors can be directly consumed by the backward
            # module, we don't need to swap out the activations.
            if (
                self.num_pipeline_stages != 1
                and self.pipeline_stage_idx != self.num_pipeline_stages - 1
            ):
                curr_module_manager.swap_out_activations(self.swap_out_stream)
            curr_module_manager.step_forward()
            # logger.debug(
            #     f"[{curr_module.name}][FWD: {curr_module_manager.finished_forward_steps}, BWD: {curr_module_manager.finished_backward_steps}] "
            #     f"Forward post hook ends.   "
            #     f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
            #     f"Peak Alloc: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB."
            # )
            return

        # (3) launch curr_module's resharding (non-blocking)
        curr_module_manager.reshard()

        # (4) launch curr_module's swap_out (non-blocking)
        self.swap_out_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.swap_out_stream):
            curr_module_manager.swap_out_weights(self.swap_out_stream)
            curr_module_manager.swap_out_activations(self.swap_out_stream)
        self.swap_out_stream.record_event()

        # (5) step the curr_module's saved tensors manager
        # this must happen after saved tensors swapping being issued
        curr_module_manager.dealloc_full_weights()

        curr_module_manager.step_forward()

        # logger.debug(
        #     f"[{curr_module.name}][FWD: {curr_module_manager.finished_forward_steps}, BWD: {curr_module_manager.finished_backward_steps}] Forward post hook ends.   "
        #     f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
        #     f"Peak Alloc: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB."
        # )

    def _backward_pre_hook(
        self,
        curr_module: nn.Module,
        next_module: nn.Module,
        module: nn.Module,
        grad_outputs: Tuple[torch.Tensor],
    ):
        """Here next_module means the next module that would be executed in the backward"""

        assert (
            module.name == curr_module.name
        ), f"Module {module.name} does not match the expected module {curr_module.name}."
        del module  # unused
        curr_module_manager = self.module_managers[curr_module.name]
        # logger.debug(
        #     f"[{curr_module.name}][FWD: {curr_module_manager.finished_forward_steps}, BWD: {curr_module_manager.finished_backward_steps}] "
        #     f"Backward pre hook begins. "
        #     f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
        #     f"Peak Alloc: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB."
        # )
        # (1) stream synchronization:
        # wait for the curr_module's swap_in
        stream_synchronize(
            self.compute_stream,
            self.swap_in_stream,
            self.unshard_stream,
        )

        if hasattr(curr_module_manager, "_saved_raw_tensors"):
            for tensor in curr_module_manager._saved_raw_tensors[curr_module_manager.finished_backward_steps]:
                tensor.record_stream(self.swap_out_stream)
            for tensor in curr_module_manager._saved_proxy_tensors[curr_module_manager.finished_backward_steps]:
                tensor.record_stream(self.swap_out_stream)

            curr_module_manager._saved_raw_tensors[curr_module_manager.finished_backward_steps] = []
            curr_module_manager._saved_proxy_tensors[curr_module_manager.finished_backward_steps] = []

        _debug_grads("Before Recover", curr_module)

        # for i, tensor in enumerate(grad_outputs):
        #     if isinstance(tensor, torch.Tensor):
        #         logger.error(
        #             f" [{curr_module.name}] PreBwd. Outputs Grad {i}: {tuple(tensor.shape)}, {tensor.float().sum().item():.6f}"
        #         )
        # for i, handle in enumerate(curr_module_manager.handles()):
        #     logger.error(
        #         f" [{curr_module.name}] PreBwd. Module Params - Before {i}: {tuple(handle.flat_param.shape)}, {handle.flat_param.float().sum().item():.6f}"
        #     )
        # for i, handle in enumerate(curr_module_manager.handles()):
        #     if handle.flat_param.grad is not None:
        #         logger.error(
        #             f" [{curr_module.name}] PreBwd. Module Grad - Before {i}: {tuple(handle.flat_param.grad.shape)}, {handle.flat_param.grad.float().sum().item():.6f}"
        #         )

        # (2) check curr_module's health
        curr_module_manager.log_flat_param_info(f"BWD Pre - Before Recover")
        self.reconstruct_full_state(
            curr_module,
            training_state=HandleTrainingState.BACKWARD_PRE,
        )
        curr_module_manager.log_flat_param_info(f"BWD Pre - After Recover")

        stream_synchronize(
            self.compute_stream,
            self.swap_in_stream,
        )

        _debug_grads("After Recover", curr_module)

        # If ALWAYS_FORWARD, this will not be triggered.
        # If ALWAYS_BACKWARD, and the next_module is None, we should backward to the
        # last module.
        # If FORWARD_BACKWARD, and being the first module, no need to swap in
        if next_module is None:
            assert curr_module.name == self.first_module_name
            if curr_module_manager.next_step_type == NextStepType.BACKWARD:
                next_module = self.modules[self.last_module_name]
            elif curr_module_manager.next_step_type == NextStepType.FORWARD:
                return

        next_module_manager = self.module_managers[next_module.name]
        next_module_manager.set_training_state(HandleTrainingState.BACKWARD_PRE)

        # (3) launch next_module's swap_in (non-blocking)
        next_module_manager.log_flat_param_info("BWD Pre - Before Next Module Swap In")
        next_module_manager.alloc_full_weights()
        with torch.cuda.stream(self.swap_in_stream):
            next_module_manager.swap_in_weights(self.swap_in_stream)
        # launch unsharding
        self.unshard_stream.wait_stream(self.swap_in_stream)
        next_module_manager.log_flat_param_info("BWD Pre - Before Next Module Unshard")
        next_module_manager.unshard(self.unshard_stream)
        next_module_manager.log_flat_param_info("BWD Pre - After Next Module Unshard")
        self.unshard_stream.record_event()
        # launch swap_in
        next_module_manager.log_flat_param_info("BWD Pre - Before Next Module Swap In")

        _debug_grads("Before Next Module Swap In", next_module)
        with torch.cuda.stream(self.swap_in_stream):
            next_module_manager.swap_in_activations(self.swap_in_stream)
            next_module_manager.log_flat_param_info("BWD Pre - After Next Module Swap In")
            self.swap_in_stream.record_event()

        with torch.cuda.stream(self.swap_out_stream):
            next_module_manager.alloc_full_grads()
            _debug_grads("After Next Module Alloc Full Grads", next_module)
            next_module_manager.swap_in_grads(self.swap_out_stream)
            _debug_grads("After Next Module Swap In Grads", next_module)

        # logger.debug(
        #     f"[{curr_module.name}][FWD: {curr_module_manager.finished_forward_steps}, BWD: {curr_module_manager.finished_backward_steps}] "
        #     f"Backward pre hook ends.   "
        #     f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB. "
        #     f"Peak Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB."
        # )

    def _backward_hook(self, curr_module: nn.Module, next_module: nn.Module, *unused):
        curr_module_manager = self.module_managers[curr_module.name]
        # logger.debug(
        #     f"[{curr_module.name}][FWD: {curr_module_manager.finished_forward_steps}, BWD: {curr_module_manager.finished_backward_steps}] "
        #     f"Backward post hook begins. "
        #     f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
        #     f"Peak Alloc: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
        # )
        curr_module_manager.set_training_state(HandleTrainingState.BACKWARD_POST)
        sharding_strategy = self.sharding_strategies[curr_module.name]
        # (1) stream synchronization:
        # wait for the prev_module's swap_out and post_backward_reduce_scatter to finish
        stream_synchronize(
            self.compute_stream,
            self.swap_out_stream,
            self.post_backward_reduce_scatter_stream,
        )

        # If ALWAYS_FORWARD, this will not be triggered.
        # If ALWAYS_BACKWARD, reshard the curr_module's params
        # If FORWARD_BACKWARD and being the first module, don't reshard
        # For non-zero and zero-1, only do the reduction for the last micro-batch.
        # For zero-2 and zero-3, do the reduction for all the micro-batches.
        # However, during pp, we do not do the overlapping because it's actually not beneficial.
        do_grad_reduce = sharding_strategy in GRAD_SHARD_HANDLE_STRATEGIES or (
            curr_module_manager.is_last_backward_micro_batch
            and parallel_state.is_pipeline_first_stage()
        )

        if (
            next_module is None
            and curr_module_manager.next_step_type == NextStepType.FORWARD
        ):
            assert curr_module.name == self.first_module_name
            # Still run the post backward grad process to move grads to the
            # correct place, reduce scatter if needed, and accumulate grads.
            curr_module_manager.reduce_grad(
                self.post_backward_reduce_scatter_stream,
                skip=not do_grad_reduce,
            )
            self.swap_out_stream.wait_stream(self.post_backward_reduce_scatter_stream)
            with torch.cuda.stream(self.swap_out_stream):
                curr_module_manager.swap_out_grads(self.swap_out_stream)
                curr_module_manager.dealloc_full_grads()
            curr_module_manager.step_backward()
            return

        # (2) launch curr_module's resharding (non-blocking)
        curr_module_manager.reshard()
        with torch.cuda.stream(self.swap_out_stream):
            curr_module_manager.swap_out_weights(self.swap_out_stream)
        curr_module_manager.dealloc_full_weights()
        curr_module_manager.reduce_grad(
            self.post_backward_reduce_scatter_stream,
            skip=not do_grad_reduce,
        )
        self.post_backward_reduce_scatter_stream.record_event()

        # (3) launch curr_module's swap_out (non-blocking)
        self.swap_out_stream.wait_stream(self.post_backward_reduce_scatter_stream)
        with torch.cuda.stream(self.swap_out_stream):
            curr_module_manager.swap_out_grads(self.swap_out_stream)
        # curr_module_manager.swap_out_grads(self.post_backward_reduce_scatter_stream)
        self.swap_out_stream.record_event()

        # (4) dealloc the full weights and grads
        curr_module_manager.dealloc_full_grads()
        curr_module_manager.step_backward()

        # logger.debug(
        #     f"[{curr_module.name}][FWD: {curr_module_manager.finished_forward_steps}, BWD: {curr_module_manager.finished_backward_steps}] "
        #     "Backward post hook ends.   "
        #     f"Curr Alloc: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
        #     f"Peak Alloc: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
        # )

    def remove_hooks(self):
        """
        Remove all the registered hooks.
        """
        for hook in self.hooks:
            hook.remove()
