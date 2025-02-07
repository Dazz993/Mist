from __future__ import annotations
from typing import (
    List,
    Optional,
    Union,
    Tuple,
    Any,
    Dict,
    Callable,
    Sequence,
    TYPE_CHECKING,
)
from functools import partial
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils._pytree import tree_flatten

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from mist.re_swap_manager.flat_param import HandleShardingStrategy, FlatParamHandle
from mist.utils.device import get_device, stream_synchronize
from mist.re_swap_manager.swap_backend import (
    get_swapped,
    swap_,
    is_tensor_on_expected_device,
    TensorSwapHandle,
)
from mist.utils.shard import shard_tensor, unshard_tensor
from mist.logger import get_logger
from mist.re_swap_manager.mem_buffer_pool import (
    RingMemoryBuffer,
    allocate_memory_buffer,
)
from mist.optimizers.fused_adam import fused_adamw_step, GradError

if TYPE_CHECKING:
    from mist.re_swap_manager.manager import (
        FlatParamGroup,
        ModuleStateSwapManager,
        ModelReSwapManager,
        ModuleReSwapManager,
    )

logger = get_logger()

name2optimizer = {
    "adamw": optim.AdamW,
}

"""
We assume the params to the ReSwapOptimizer are the FlatHandle for FlatParameters.
the grads are saved in _saved_grad, and we make sure the _saved_grad is 
always correct (being reduce-scattered) when we call step().

If sharding_strategy == NO_SHARD:
    - If fully_cuda: step
    - If fully_cpu: swap main params, grads, state tensors to cpu and compute on cpu
    - If partial: swap main params, grads, state tensors to cuda and compute on cuda

If sharding_strategy == OPT_ONLY:
    PRE-CONDITION: the grads are reduce-scatterred in the last iteration. and saved in _saved
    - If fully_cuda: sharded main params, sharded grads, and sharded state tensors are in cuda, compute on cuda
    - If fully_cpu: swap sharded main params, grads, state tensors to cpu and compute on cpu
    - If partial: swap sharded main params, grads, state tensors to cuda and compute on cuda

If sharding_strategy == OPT_AND_GRAD:
    PRE_CONDITION: the grads are reduce-scatterred

"""


def _multi_tensor_copy_this_to_that(this, that, overflow_buf=None):
    """Use multi-tensor-applier to copy values from one list to another.
    We don't have a blfoat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16."""
    if overflow_buf:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_, non_blocking=True)


def _init_state(state, param_group, main_param, device):
    assert len(state) == 0, "Optimizer states should be empty"
    if "capturable" not in param_group:
        param_group["capturable"] = False
    if "fused" not in param_group:
        param_group["fused"] = False
    if "amsgrad" not in param_group:
        param_group["amsgrad"] = False
    state["step"] = (
        torch.zeros((), dtype=torch.float, device=device)
        if param_group["capturable"] or param_group["fused"]
        else torch.tensor(0.0)
    )
    # Exponential moving average of gradient values
    state["exp_avg"] = torch.zeros_like(
        main_param,
        memory_format=torch.preserve_format,
        device=device,
    )
    # Exponential moving average of squared gradient values
    state["exp_avg_sq"] = torch.zeros_like(
        main_param,
        memory_format=torch.preserve_format,
        device=device,
    )
    if param_group["amsgrad"]:
        # Maintains max of all exp. moving avg. of sq. grad. values
        state["max_exp_avg_sq"] = torch.zeros_like(
            main_param,
            memory_format=torch.preserve_format,
            device=device,
        )


class ModuleOptimizerHandle:
    def __init__(
        self,
        module: nn.Module,
        optimizer_cls: optim.Optimizer,
        optimizer_kwargs: Dict[str, Any],
        module_manager: ModuleReSwapManager,
        process_group: Optional[dist.ProcessGroup] = None,
        swap_ratio: float = 0.0,
        sharding_strategy: HandleShardingStrategy = HandleShardingStrategy.NO_SHARD,
        cpu_optim_step: bool = False,
        grad_scaler: Optional[ShardedGradScaler] = None,
    ):
        self.module = module
        self.module_name = module.name
        self.module_manager = module_manager
        # Create params and the optimizer
        self.params_to_handle = {}
        for p in module_manager.handles(requires_grad=True):
            self.params_to_handle[p.flat_param] = p
        self.optimizer = optimizer_cls(
            params=self.params_to_handle.keys(), **optimizer_kwargs
        )

        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        self.swap_ratio = swap_ratio
        self.sharding_strategy = sharding_strategy
        self.cpu_optim_step = cpu_optim_step
        self.grad_scaler = grad_scaler
        self.default_inv_scale = torch.ones((1,), device="cuda")

        if self.cpu_optim_step and swap_ratio != 1.0:
            raise ValueError(
                f"When compute device is cpu, swap ratio must be 1.0, got {swap_ratio}"
            )
        self.compute_device = (
            torch.device("cpu")
            if self.cpu_optim_step
            else get_device(torch.cuda.current_device())
        )
        self.fully_cpu = self.cpu_optim_step
        self.fully_cuda = self.swap_ratio == 0.0

        self.ready_to_step = False

        # Swap in weights if needed
        self.module_manager.alloc_full_weights()
        self.module_manager.swap_in_weights(torch.cuda.current_stream())

        self.module_manager.log_flat_param_info(f"[1] DEBUG IN OPTIMIZER INIT")

        # Terminology:
        #   main param: the param that the optimizer is using
        #   model param: the param that the model is using
        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []

        for i, param_group in enumerate(self.optimizer.param_groups):
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            for i, param in enumerate(param_group["params"]):
                if not param.requires_grad:
                    continue

                main_param_needs_sharding = self.sharding_strategy in (
                    HandleShardingStrategy.OPT_ONLY,
                    HandleShardingStrategy.OPT_AND_GRAD,
                )

                # Float16 params
                if param.dtype in (torch.float16, torch.bfloat16):
                    float16_params_this_group.append(param)
                    main_param = param.detach().clone().float()

                    # Shard the main param if OPT is sharded but param is not
                    if main_param_needs_sharding:
                        main_param = shard_tensor(
                            main_param,
                            self.world_size,
                            self.rank,
                            dim=0,
                            share_storage=False,
                        )
                        main_param._is_sharded = True

                    # Replace the optimizer params with the fp32 copy
                    param_group["params"][i] = main_param
                    fp32_from_float16_params_this_group.append(main_param)
                    # Reset existing state dict key to the main param
                    if param in self.optimizer.state:
                        self.optimizer.state[main_param] = self.optimizer.state.pop(
                            param
                        )
                    state = self.optimizer.state[main_param]

                # FP32 params
                elif param.dtype == torch.float32:
                    fp32_params_this_group.append(param)
                    param_group["params"][i] = param
                    state = self.optimizer.state[param]

                    # Shard the param if needed to generate the correct state partition
                    if main_param_needs_sharding:
                        main_param = shard_tensor(
                            param,
                            self.world_size,
                            self.rank,
                            dim=0,
                            share_storage=True,
                        )

                else:
                    raise ValueError(f"Unknown param dtype: {param.dtype}")

                # ================================================================
                # We actively init the optimizer states
                _init_state(state, param_group, main_param, self.compute_device)
                self.module_manager.log_flat_param_info(f"[2] DEBUG IN OPTIMIZER INIT")
                # ================================================================

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # Init the swapped params and states
        self.fp32_params_list = tree_flatten(self.fp32_from_float16_groups)[0]
        self.fp32_params_numel_list = [p.numel() for p in self.fp32_params_list]
        self.opt_states_list = []
        for param, state in self.optimizer.state.items():
            self.opt_states_list.append(state["exp_avg"])
            self.opt_states_list.append(state["exp_avg_sq"])
        self.opt_states_numel_list = [p.numel() for p in self.opt_states_list]

        # Update the optimizer step to use the fused adamw step
        self.optimizer.step = partial(
            fused_adamw_step,
            states=self.optimizer.state,
            param_groups=self.optimizer.param_groups,
            # capturable=True,
            message=f"{self.module_name} Step",
        )

        # Init the swapping handles
        curr_stream = torch.cuda.current_stream()
        self.postprocess_copy_main_params_to_model_params(curr_stream)
        self.postprocess_swap_out(curr_stream, zero_grad=True)
        # for tensor in self.fp32_params_list:
        #     swap_(tensor, state="cuda", stream=curr_stream)
        # for tensor in self.opt_states_list:
        #     swap_(tensor, state="cuda", stream=curr_stream)
        # torch.cuda.synchronize()
        # self.postprocess_swap_out(curr_stream, zero_grad=True)
        # torch.cuda.synchronize()
        # # ================================================================
        # # Debug print
        # for tensor in self.fp32_params_list:
        #     if hasattr(tensor, "_cuda_data"):
        #         print(
        #             f"[{self.module_name}] CUDA Data: {tensor._cuda_data.sum().item():.4f}, CPU Data: {tensor._cpu_data.sum().item():.4f}"
        #         )
        # for tensor in self.opt_states_list:
        #     if hasattr(tensor, "_cuda_data"):
        #         print(
        #             f"[{self.module_name}] CUDA Data: {tensor._cuda_data.sum().item():.4f}, CPU Data: {tensor._cpu_data.sum().item():.4f}"
        #         )
        # # ================================================================
        self.module_manager.swap_out_weights(curr_stream)
        self.module_manager.dealloc_full_weights()

    def preprocess(
        self,
        stream: torch.cuda.Stream,
        cuda_buffers: Optional[List[torch.Tensor]] = None,
    ):
        """Check healthy and if not healthy, swap in"""
        if self.cpu_optim_step:
            state = "cpu"
        elif self.compute_device.type == "cuda":
            state = "cuda"
        else:
            raise ValueError(f"Unknown compute device: {self.compute_device.type}")

        if cuda_buffers is not None and cuda_buffers:
            assert len(cuda_buffers) == len(self.fp32_params_list) + len(
                self.opt_states_list
            ), (
                f"cuda_buffers length: {len(cuda_buffers)}, "
                f"fp32_params_list length: {len(self.fp32_params_list)}, "
                f"opt_states_list length: {len(self.opt_states_list)}"
            )
        if cuda_buffers is None or not cuda_buffers:
            cuda_buffers = [None] * (
                len(self.fp32_params_list) + len(self.opt_states_list)
            )

        with torch.profiler.record_function(f"OPT-{self.module_name}.preprocess"):
            with torch.profiler.record_function(f"grads_swap_{state}"):
                for handle in self.module_manager.handles(requires_grad=True):
                    if handle.grad_swap_ratio == 0.0:
                        continue
                    handle.alloc_full_grads()
                    handle.swap_in_grads(
                        stream=stream, shard=handle.uses_sharded_strategy
                    )

            opt_states = [*self.fp32_params_list, *self.opt_states_list]

            with torch.profiler.record_function(f"state_swap_{state}"):
                for tensor, cuda_buffer in zip(opt_states, cuda_buffers):
                    # if not hasattr(tensor, "_cpu_data"):
                    #     logger.debug(
                    #         f"[Opt-Preprocess] [{self.module_name}] No _cpu_data"
                    #     )
                    # else:
                    #     # print(
                    #     #     f"[111222] [{self.module_name}] tensor._cpu_data.sum().item(): {tensor._cpu_data.sum().item():.4f}, tensor._cuda_data.sum().item(): {tensor._cuda_data.sum().item():.4f}"
                    #     # )
                    #     logger.debug(
                    #         f"[Opt-Preprocess] [{self.module_name}] Get some value: {tensor._cuda_data[0]:.4f}"
                    #     )
                    if hasattr(tensor, "_swap_handle"):
                        if cuda_buffer is not None:
                            tensor._ori_cuda_data_as_a_buffer = tensor._cuda_data
                        swap_(
                            tensor, state=state, stream=stream, cuda_buffer=cuda_buffer
                        )
                        TensorSwapHandle.clean(tensor)

            # torch.cuda.synchronize()

            with torch.profiler.record_function(f"model_grads_to_main_grads"):
                with torch.cuda.stream(stream):
                    self._copy_model_grads_to_main_grads()

    def step(self, closure: Optional[Callable] = None):
        # logger.debug(
        #     f"[{self.module_name}] Is FlatParam: {getattr(self.float16_groups[0][0], '_is_flat_param', False)}"
        # )

        if self.grad_scaler is not None:
            inv_scale = self.grad_scaler.inv_scale
        else:
            inv_scale = self.default_inv_scale

        with torch.profiler.record_function(f"OPT-{self.module_name}.step"):
            try:
                self.optimizer.step(closure=closure, inv_scale=inv_scale)
            except GradError as e:
                logger.error(f"GradError: {e} happened in {self.module_name}")

        # logger.debug(
        #     f"[{self.module_name}] Is FlatParam: {getattr(self.float16_groups[0][0], '_is_flat_param', False)}"
        # )

    def postprocess_swap_out(self, stream: torch.cuda.Stream, zero_grad: bool = True):
        """Postprocess after step
        1. if original param is in cpu and compute device is gpu, swap out the fp32 copy
        2. update the fp16 params from fp32 copy
        Not: free gradients (this should be called explicitly by the user via `zero_grad`)
        """

        # logger.error(f"[{self.module_name}] Postprocess Swap Out")

        # logger.debug(
        #     f"[{self.module_name}] Is FlatParam: {getattr(self.float16_groups[0][0], '_is_flat_param', False)}"
        # )

        state = "partial"

        with torch.profiler.record_function(f"OPT-{self.module_name}.postprocess"):
            for handle in self.module_manager.handles(requires_grad=True):
                handle.dealloc_full_grads()

            with torch.profiler.record_function(f"zero_grad"):
                if zero_grad:
                    self.zero_grad()

            # logger.info(
            #     f"[1] Memory Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            # )

            opt_states = [*self.fp32_params_list, *self.opt_states_list]
            numel_list = [*self.fp32_params_numel_list, *self.opt_states_numel_list]

            with torch.profiler.record_function(f"state_swap_{state}"):
                for tensor, numel in zip(opt_states, numel_list):
                    # logger.error(
                    #     f"[{self.module_name}] Swap out tensor. numel: {numel}. dst_numel_in_cuda_for_partial: {int(numel * (1 - self.swap_ratio))}"
                    # )
                    cuda_buffer = None
                    if hasattr(tensor, "_ori_cuda_data_as_a_buffer"):
                        cuda_buffer = tensor._ori_cuda_data_as_a_buffer
                    swap_(
                        tensor,
                        state=state,
                        dst_numel_in_cuda_for_partial=int(
                            numel * (1 - self.swap_ratio)
                        ),
                        stream=stream,
                        cuda_buffer=cuda_buffer,
                    )
                    if hasattr(tensor, "_ori_cuda_data_as_a_buffer"):
                        del tensor._ori_cuda_data_as_a_buffer

            # logger.info(
            #     f"[2] Memory Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            # )

            # logger.debug(
            #     f"[{self.module_name}] Is FlatParam: {getattr(self.float16_groups[0][0], '_is_flat_param', False)}"
            # )

    def postprocess_copy_main_params_to_model_params(self, stream: torch.cuda.Stream):

        # logger.error(
        #     f"[{self.module_name}] Postprocess Copy Main Params to Model Params"
        # )

        with torch.profiler.record_function(f"OPT-{self.module_name}.postprocess"):
            with torch.profiler.record_function(f"main_params_to_model_params"):
                with torch.cuda.stream(stream):
                    self._copy_main_params_to_model_params(stream)

        # logger.debug(
        #     f"[{self.module_name}] Is FlatParam: {getattr(self.float16_groups[0][0], '_is_flat_param', False)}"
        # )

    def zero_grad(self, set_to_none: bool = False):
        """Free gradients"""
        # Delete extra main grads
        for fp32_group in self.fp32_from_float16_groups:
            for fp32_param in fp32_group:
                fp32_param.grad = None
        # Delete other grads
        for fp32_group in self.fp32_from_fp32_groups:
            for fp32_param in fp32_group:
                fp32_param.grad = None
        for fp16_group in self.float16_groups:
            for fp16_param in fp16_group:
                fp16_param.grad = None

        # Reset swapped grads
        self.module_manager.zero_grad()

    def _debug_compute_main_params(self, message: str = ""):
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                print(f"[{message=}] Main Param: {main_param.float().sum().item():.4f}")

    def _get_model_and_main_params_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(
            self.float16_groups, self.fp32_from_float16_groups
        ):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param)
                main_data.append(main_param)

        logger.debug(
            f"[{self.module_name}] Is FlatParam: {getattr(self.float16_groups[0][0], '_is_flat_param', False)}"
        )
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        # After grads are swapped to the correct device, we need to assign the
        # grads to the main (fp32) params
        for fp16_group, fp32_group in zip(
            self.float16_groups, self.fp32_from_float16_groups
        ):
            for fp16_param, fp32_param in zip(fp16_group, fp32_group):
                assert hasattr(fp16_param, "_saved_grad")
                # Shard the grads if needed. Because fp32_param may be sharded
                # while the grads are not sharded
                grad = fp16_param._saved_grad
                assert grad.numel() == fp32_param.numel(), (
                    f"[Module] {self.module_name}: "
                    f"grad numel: {grad.numel()}, "
                    f"fp32_param numel: {fp32_param.numel()}"
                )
                fp32_param.grad = torch.empty(
                    grad.size(), dtype=torch.float, device=grad.device
                )
                # fp32_param.grad.copy_(grad, non_blocking=True)
                fp32_param.grad = grad.float()

    def _copy_main_params_to_model_params(self, stream: torch.cuda.Stream):
        # Only needed for the float16 params.

        logger.debug(
            f"[{self.module_name}] Is FlatParam: {getattr(self.float16_groups[0][0], '_is_flat_param', False)}"
        )

        model_data, main_data = self._get_model_and_main_params_float16()
        flat_params = model_data
        fp32_params = main_data
        # model_data[i] is the flat_param
        # Unshard the main params if needed
        # (1) the param is unsharded in the forward backward and still not resharded
        # (2) the strategy is OPT_ONLY or OPT_AND_GRAD
        for i in range(len(fp32_params)):
            flat_param = flat_params[i]
            fp32_param = fp32_params[i]
            # print(
            #     f"[DEBUG - Copy main params to model params - 1] {self.module_name}, fp32_param: {fp32_param.float().sum().item():.8f}, flat_param: {flat_param.float().sum().item():.8f}"
            # )
            assert getattr(flat_param, "_is_flat_param", False), (
                f"[Module] {self.module_name}: "
                f"model_data[i]: {flat_param}, "
                f"main_data[i]: {fp32_param}"
            )
            # Clean the local shard swapping info because we will directly copy
            # from fp32 weight to fp16 weight
            TensorSwapHandle.clean(flat_param._local_shard)

            sharding_strategy = self.sharding_strategy
            flat_param_full_size = flat_param._unsharded_size
            flat_param_sharded_size = flat_param._sharded_size
            fp32_param_size = fp32_param.size()

            if sharding_strategy in (
                HandleShardingStrategy.OPT_ONLY,
                HandleShardingStrategy.OPT_AND_GRAD,
            ):
                # Opt states are sharded while the model params are not
                # In this case, we need to all gather to update the model params
                assert fp32_param_size == flat_param_sharded_size, (
                    f"[Module] {self.module_name}: "
                    f"fp32_param_size.size: {fp32_param_size.size}, "
                    f"flat_param_sharded_size: {flat_param_sharded_size}"
                )
                # Point local shard to full param
                flat_param._local_shard.set_(flat_param._full_param)
                # Unshard to update the full param
                start = fp32_param_size.numel() * self.rank
                end = start + fp32_param_size.numel()
                partition = flat_param._full_param.view(-1)[start:end]
                partition.data.copy_(fp32_param.view(-1))
                partition = partition.view(fp32_param_size)
                # stream.wait_stream(torch.cuda.current_stream())
                # print(
                #     f"[DEBUG - Copy main params to model params] {self.module_name} {partition.float().sum().item():.4f}, {flat_param._full_param.float().sum().item():.4f}"
                # )
                unshard_tensor(
                    shard=partition,
                    num_shards=self.world_size,
                    process_group=self.process_group,
                    dim=0,
                    buffer=flat_param._full_param,
                    stream=stream,
                    async_op=False,
                )
                # torch.cuda.synchronize()
                # print(
                #     f"[DEBUG - After unshard: {self.module_name} {partition.float().sum().item():.4f}, {flat_param._full_param.float().sum().item():.4f}"
                # )
            elif sharding_strategy == HandleShardingStrategy.NO_SHARD:
                flat_param._full_param.data.copy_(fp32_param)
                flat_param._local_shard.set_(flat_param._full_param)
            else:  # sharding_strategy == HandleShardingStrategy.FULL_SHARD
                # If both sharded, we can directly copy, becasue all gathering
                # will happen in the forward.
                # flat_param._local_shard.data = flat_param._full_param[
                #     : fp32_param_size.numel()
                # ]
                flat_param._local_shard.data.copy_(fp32_param.view(-1))

            flat_param.data = flat_param._local_shard

    # def _copy_model_params_to_main_params(self, stream: torch.cuda.Stream):
    #     raise NotImplementedError(f"Copy model params to main params not implemented")


class ReSwapAdamW:
    def __init__(
        self,
        re_swap_manager: ModelReSwapManager,
    ):
        # Root model and submodules
        self.root_model: nn.Module = re_swap_manager.model
        self.modules: Dict[str, nn.Module] = re_swap_manager.modules

        # Optimizer handles
        module_optimizers = re_swap_manager.module_optimizers
        self.module_optimizer_handles: List[ModuleOptimizerHandle] = [
            v for k, v in module_optimizers.items()
        ]

        # Grad scaler
        self.grad_scaler = re_swap_manager.grad_scaler

        # Stream
        # The update can only be
        # 1. param originally is in cpu, and compute device is gpu
        # 2. param originally is in cpu, and compute device is cpu
        # 3. param originally is in gpu, and compute device is gpu
        self.compute_stream = torch.cuda.current_stream()
        # Swap in is only needed if param is in cpu and the compute device is gpu
        self.swap_in_stream = torch.cuda.Stream()
        self.post_copy_stream = torch.cuda.Stream()

    def _create_full_memory_buffers(self):
        pass

    def step(self, zero_grad=True, closure: Optional[Callable] = None):
        """
        Step with mixed precision (combined with sharding and offloading)
        1. Step with root model params (no swapping)
        2. For per module params, step with overlapping
            - Precondition: has fp16 params, fp32 copy of fp16 params, fp16/fp32 grads
            - 1) If not in the compute device, copy to the compute device (check healthy)
            - 2) Run step
            - 3) Update fp16 params
            - 4) Postprocess
        """

        def get_optimizer_handle(index: int) -> ModuleOptimizerHandle:
            if index >= len(self.module_optimizer_handles):
                return None
            return self.module_optimizer_handles[index]

        # Prefetch next optimizer if it needs to be swapped in
        next_optimizer_handle = get_optimizer_handle(0)
        if next_optimizer_handle is not None:
            next_optimizer_handle.preprocess(self.swap_in_stream)

        for i, optimizer_handle in enumerate(self.module_optimizer_handles):
            curr_optimizer_handle = optimizer_handle
            next_optimizer_handle = get_optimizer_handle(i + 1)

            # ##########
            # Pre-step #
            # ##########
            # Make sure the current optimizer is healthy
            self.compute_stream.wait_stream(self.swap_in_stream)
            self.swap_in_stream.wait_stream(self.compute_stream)

            if next_optimizer_handle is not None:
                # Swap in next optimizer if needed
                next_optimizer_handle.preprocess(self.swap_in_stream)

            # ######
            # Step #
            # ######
            curr_optimizer_handle.step(closure=closure)

            # ###########
            # Post-step #
            # ###########
            self.post_copy_stream.wait_stream(self.compute_stream)
            curr_optimizer_handle.postprocess(
                zero_grad=zero_grad, stream=self.post_copy_stream
            )

        torch.cuda.current_stream().record_event()
        torch.cuda.synchronize()

        if self.grad_scaler is not None:
            self.grad_scaler.update()

    def zero_grad(self, set_to_none: bool = False):
        for optimizer_handle in self.module_optimizer_handles:
            optimizer_handle.zero_grad(set_to_none=set_to_none)
