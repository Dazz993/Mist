# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modified by Zhanda Zhu, 2023.
from __future__ import annotations
import contextlib
from typing import Callable, Iterator, List, Optional, Union, Tuple

import torch
import torch.distributed
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from mist.config import MistConfig
from mist import parallel_state
from mist.logger import get_logger
from mist.re_swap_manager.manager import ModelReSwapManager

# from dazzle.core.enums import ModelType
# from dazzle.pipeline_parallel import p2p_communication
# from dazzle.core.utils import get_attr_wrapped_model, get_model_config, get_model_type

logger = get_logger(__name__)

# Types
Shape = Union[List[int], torch.Size]


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    """
    pipeline_parallel_size = parallel_state.get_num_pipeline_stages()
    if pipeline_parallel_size > 1:
        forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(output_tensors, deallocate_pipeline_outputs=False):
    """Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    """
    if (output_tensors is None) or (not deallocate_pipeline_outputs):
        return
    if isinstance(output_tensors, torch.Tensor):
        output_tensors = [output_tensors]
    for out in output_tensors:
        assert isinstance(out, torch.Tensor), (
            "expected Tensor, found %s." % type(out).__name__
        )
        assert (
            out._base is None
        ), f"counter-productive to free a view of another tensor. Rank: {torch.distributed.get_rank()}. Tensor.shape: {out.shape}. Tensor.base: {out._base.shape}."
        out.data = torch.empty(
            (1,),
            device=out.device,
            dtype=out.dtype,
        )


def custom_backward(output, grad_output):
    """Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    """

    assert (
        output.numel() == 1
    ), f"output should be pseudo-'freed' in schedule, to optimize memory. {output.shape=}"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format=torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensors,
    forward_data_store,
    config,
):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensors is used.

    Returns output tensor."""
    if not isinstance(input_tensors, (tuple, list)):
        input_tensors = [input_tensors]

    # Set the input tensors.
    model.input_tensors = input_tensors

    if getattr(config, "enable_autocast", False):
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        output_tensors = forward_step_func(data_iterator, model)

    if parallel_state.is_pipeline_last_stage():
        loss = output_tensors
        output_tensors = loss / num_microbatches
        forward_data_store.append(loss.detach())

    # Reset the input tensors.
    model.input_tensors = None

    logger.debug(
        f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
        f"Finished forward pass. "
    )

    return output_tensors


def backward_step(
    input_tensors,
    output_tensors,
    output_tensor_grads,
    config,
    deallocate_pipeline_outputs=False,
):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # Retain the grad on the input_tensors.
    unwrap_input_tensor_grads = False
    if not isinstance(input_tensors, (tuple, list)):
        input_tensors = [input_tensors]
        unwrap_input_tensor_grads = True
    for x in input_tensors:
        if x is not None and x.requires_grad:
            x.retain_grad()

    if not isinstance(output_tensors, (tuple, list)):
        output_tensors = [output_tensors]
    if not isinstance(output_tensor_grads, (tuple, list)):
        output_tensor_grads = [output_tensor_grads]

    # Get the output tensor grads.
    indices = []
    for i, (tensor, grad) in enumerate(zip(output_tensors, output_tensor_grads)):
        if tensor.requires_grad and tensor.grad_fn is not None:
            indices.append(i)
    assert (
        len(indices) == 1
    ), f"only one tensor should require grad, otherwise need further logic, got {indices}"
    idx = indices[0]

    # Backward pass.
    if (
        output_tensor_grads[idx] is None
        and getattr(config, "grad_scalar", None) is not None
    ):
        output_tensors[idx] = config.grad_scalar.scale(output_tensors[idx])

    if deallocate_pipeline_outputs:
        custom_backward(output_tensors[idx], output_tensor_grads[idx])
    else:
        torch.autograd.backward(
            output_tensors[idx], grad_tensors=output_tensor_grads[idx]
        )

    # Collect the grad of the input_tensors.
    input_tensor_grad = [None]
    if input_tensors is not None:
        input_tensor_grad = []
        for x in input_tensors:
            if x is None or not x.requires_grad:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grads:
        input_tensor_grad = input_tensor_grad[0]

    logger.debug(
        f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
        f"Finished backward pass. "
    )

    return input_tensor_grad


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    model_re_swap_manager: ModelReSwapManager,
    config: MistConfig,
    forward_only: bool = False,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.
    """

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    forward_data_store = []
    input_tensors, output_tensor_grad = None, None
    num_microbatches = config.training.gradient_accumulation_steps
    for i in range(num_microbatches):
        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensors,
            forward_data_store,
            config,
        )
        if not forward_only:
            backward_step(input_tensors, output_tensor, output_tensor_grad, config)

    return forward_data_store


def get_tensor_shapes(
    *,
    rank: int,
    model_type,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).

    if config.sequence_parallel:
        raise NotImplementedError(
            "Sequence parallelism not yet supported in non-interleaved pipeline parallelism"
        )

    assert (
        config.send_recv_tensor_property is not None
    ), "send_recv_tensor_property must be set in config for non-interleaved pipeline parallelism"

    return config.send_recv_tensor_property


def recv(
    tensor_properties,
    ranks: List[int],
    shard_info: Optional[parallel_state.ShardInfo] = None,
    group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Receive tensors from ranks given tensor properties and shard info."""
    if len(ranks) == 0:
        return [], []
    if shard_info is None:
        shard_info = parallel_state.ShardInfo(False)
    if shard_info.is_sharded and (len(ranks) != shard_info.num_shards):
        raise ValueError(
            f"Number of ranks ({len(ranks)}) must equal number of shards ({shard_info.num_shards})"
        )
    if not shard_info.is_sharded and len(ranks) != 1:
        raise ValueError(
            f"Number of ranks ({len(ranks)}) must equal 1 for non-sharded tensors"
        )

    ranks = sorted(ranks)
    num_shards = shard_info.num_shards
    input_tensors = []
    ops = []
    for i, tensor_property in enumerate(tensor_properties):
        if tensor_property is None:
            input_tensors.append(None)
            continue
        # Received tensor is sharded, and the final tensor should be concatenated
        shape, dtype, requires_grad = tensor_property
        shape = list(shape)
        orig_dim_0 = shape[0]
        shape[0] = shape[0] * num_shards
        input_tensor = torch.empty(
            shape,
            dtype=dtype,
            requires_grad=requires_grad,
            device=torch.cuda.current_device(),
        )
        input_tensors.append(input_tensor)
        for shard_idx, rank in enumerate(ranks):
            start = shard_idx * orig_dim_0
            end = (shard_idx + 1) * orig_dim_0
            tensor_to_recv = input_tensor[start:end]
            op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor=input_tensor[start:end],
                peer=rank,
                group=group,
            )
            ops.append(op)
            logger.debug(
                f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
                f"Begin to receive tensor with "
                f"(Shape: {tuple(tensor_to_recv.shape)}) "
                f"(Dtype: {tensor_to_recv.dtype}) "
                f"(Requires_grad: {tensor_to_recv.requires_grad}) "
                f"from rank {rank}"
            )

    return input_tensors, ops


def send(
    tensors: List[Optional[torch.Tensor]],
    ranks: List[int],
    shard_info: parallel_state.ShardInfo,
    group: torch.distributed.ProcessGroup,
):
    """Receive tensors from ranks given tensor properties and shard info."""
    if len(ranks) == 0:
        return None

    ranks = sorted(ranks)
    num_shards = shard_info.num_shards
    ops = []
    for i, tensor in enumerate(tensors):
        if tensor is None:
            continue
        assert isinstance(
            tensor, torch.Tensor
        ), f"tensor at index {i} is not a torch.Tensor"
        shape = list(tensor.shape)
        assert shape[0] % num_shards == 0
        sharded_dim = shape[0] // num_shards
        for shard_idx, rank in enumerate(ranks):
            start = shard_idx * sharded_dim
            end = (shard_idx + 1) * sharded_dim
            shard = tensor[start:end]
            op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor=shard,
                peer=rank,
                group=group,
            )
            ops.append(op)
            logger.debug(
                f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
                f"Begin to send tensor with "
                f"(Shape: {tuple(shard.shape)}) "
                f"(Dtype: {shard.dtype}) "
                f"(Requires_grad: {shard.requires_grad}) "
                f"to rank {rank}"
            )

    return ops


def recv_forward(p2p_handler: P2PHandler, issue: bool = True):
    """Receive forward tensors from previous pipeline stage."""
    if parallel_state.is_pipeline_first_stage():
        return None
    if p2p_handler.recv_forward_tensor_properties is None:
        p2p_handler.init_recv_forward()

    recv_infos = p2p_handler.pp_forward_recv_prev_infos
    shard_info: parallel_state.ShardInfo = recv_infos[0][1]
    ranks: List[int] = [rank for rank, _ in recv_infos]
    tensor_properties = p2p_handler.recv_forward_tensor_properties

    logger.debug(
        f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
        f"Begin to receive forward tensors from previous stage. "
        f"Tensor properties: {tensor_properties}"
    )

    input_tensors, ops = recv(
        tensor_properties,
        ranks,
        shard_info=shard_info,
        group=p2p_handler.group,
    )

    if issue and len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        logger.debug(
            f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
            f"Finished receiving forward tensors from previous stage. "
            f"Number of tensors: {len(input_tensors)}"
        )
        torch.cuda.synchronize()
    elif not issue:
        return input_tensors, ops

    return input_tensors


def recv_backward(p2p_handler: P2PHandler, issue: bool = True):
    """Receive backward tensors from next pipeline stage."""
    if parallel_state.is_pipeline_last_stage():
        return None
    if p2p_handler.recv_backward_tensor_properties is None:
        p2p_handler.init_recv_backward()

    recv_infos = p2p_handler.pp_backward_recv_next_infos
    shard_info: parallel_state.ShardInfo = recv_infos[0][1]
    ranks: List[int] = [rank for rank, _ in recv_infos]
    tensor_properties = p2p_handler.recv_backward_tensor_properties

    logger.debug(
        f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
        f"Begin to receive backward tensors from next stage. "
        f"Tensor properties: {tensor_properties}"
    )

    input_tensors, ops = recv(
        tensor_properties,
        ranks,
        shard_info=shard_info,
        group=p2p_handler.group,
    )

    if issue and len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        logger.debug(
            f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
            f"Finished receiving backward tensors from next stage. "
            f"Number of tensors: {len(input_tensors)}"
        )
        torch.cuda.synchronize()
    elif not issue:
        return input_tensors, ops

    return input_tensors


def send_forward(tensors, p2p_handler: P2PHandler, issue: bool = True):
    """Send forward tensors to next pipeline stage."""
    if parallel_state.is_pipeline_last_stage():
        return
    if not p2p_handler.send_forward_initialized:
        p2p_handler.init_send_forward(tensors)

    send_infos = p2p_handler.pp_forward_send_next_infos
    shard_info: parallel_state.ShardInfo = send_infos[0][1]
    ranks: List[int] = [rank for rank, _ in send_infos]

    logger.debug(
        f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
        f"Begin to send forward tensors to next stage. "
    )

    ops = send(
        tensors,
        ranks,
        shard_info=shard_info,
        group=p2p_handler.group,
    )

    if issue and len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        logger.debug(
            f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
            f"Finished sending forward tensors to next stage. "
        )
        torch.cuda.synchronize()
    elif not issue:
        return ops


def send_backward(tensors, p2p_handler: P2PHandler, issue: bool = True):
    if parallel_state.is_pipeline_first_stage():
        return
    if not p2p_handler.send_backward_initialized:
        p2p_handler.init_send_backward(tensors)

    send_infos = p2p_handler.pp_backward_send_prev_infos
    shard_info: parallel_state.ShardInfo = send_infos[0][1]
    ranks: List[int] = [rank for rank, _ in send_infos]

    logger.debug(
        f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
        f"Begin to send backward tensors to previous stage. "
    )

    ops = send(
        tensors,
        ranks,
        shard_info=shard_info,
        group=p2p_handler.group,
    )

    if issue and len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        logger.debug(
            f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
            f"Finished sending backward tensors to previous stage. "
        )
        torch.cuda.synchronize()
    elif not issue:
        return ops


def send_forward_recv_backward(tensors, p2p_handler: P2PHandler):
    if parallel_state.is_pipeline_last_stage():
        return None
    send_ops = send_forward(tensors, p2p_handler, issue=False)
    input_tensor_grads, recv_ops = recv_backward(p2p_handler, issue=False)
    ops = send_ops + recv_ops
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        logger.debug(
            f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
            f"Finished sending forward tensors to next stage and receiving backward tensors from previous stage. "
        )
        torch.cuda.synchronize()
    return input_tensor_grads


def send_backward_recv_forward(tensors, p2p_handler: P2PHandler):
    if parallel_state.is_pipeline_first_stage():
        return None
    send_ops = send_backward(tensors, p2p_handler, issue=False)
    input_tensors, recv_ops = recv_forward(p2p_handler, issue=False)
    ops = send_ops + recv_ops
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        logger.debug(
            f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
            f"Finished sending backward tensors to previous stage and receiving forward tensors from next stage. "
        )
        torch.cuda.synchronize()
    return input_tensors


DTYPE_TO_INT = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.int16: 5,
    torch.int8: 6,
    torch.uint8: 7,
    torch.bool: 8,
}
INT_TO_DTYPE = {v: k for k, v in DTYPE_TO_INT.items()}


class P2PHandler:
    MAX_TENSORS_TO_COMM = 10
    MAX_TENSOR_NDIM = 6
    TENSOR_METADATA_DTYPE = torch.int32

    def __init__(self, config: MistConfig):
        self.config = config
        self.num_stages = parallel_state.get_num_pipeline_stages()
        self.stage_idx = parallel_state.get_pipeline_parallel_stage_idx()
        self.pp_forward_recv_prev_infos: List[Tuple[int, parallel_state.ShardInfo]] = (
            parallel_state.get_pipeline_parallel_forward_recv_prev_infos()
        )
        self.pp_forward_send_next_infos: List[Tuple[int, parallel_state.ShardInfo]] = (
            parallel_state.get_pipeline_parallel_forward_send_next_infos()
        )
        self.pp_backward_recv_next_infos: List[Tuple[int, parallel_state.ShardInfo]] = (
            parallel_state.get_pipeline_parallel_backward_recv_next_rank()
        )
        self.pp_backward_send_prev_infos: List[Tuple[int, parallel_state.ShardInfo]] = (
            parallel_state.get_pipeline_parallel_backward_send_prev_ranks()
        )
        # The group is the WORLD group because we directly get the recv and next
        # ranks from the global group.
        self.group = torch.distributed.group.WORLD
        # States
        self.send_forward_initialized = False
        self.send_backward_initialized = False
        self.recv_forward_tensor_properties = None
        self.recv_backward_tensor_properties = None

    def init_send_forward(self, tensors: List[Optional[torch.Tensor]]):
        if self.send_forward_initialized:
            return
        logger.debug(
            f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
            f"Begins initializing send forward for stage {self.stage_idx}."
        )
        if len(self.pp_forward_send_next_infos) != 0:
            shard_info = self.pp_forward_send_next_infos[0][1]
            assert set(
                info.num_shards for _, info in self.pp_forward_send_next_infos
            ) == {
                shard_info.num_shards
            }, f"All tensors must have the same shard info, got {self.pp_forward_send_next_infos}"
            metadata = self._encode_tensor_metadata(tensors)
            metadata = self._update_tensor_metadata(metadata, shard_info)
            for rank, _ in self.pp_forward_send_next_infos:
                logger.debug(
                    f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
                    f"Sending tensor metadata to rank {rank} for stage {self.stage_idx}, "
                    f"metadata shape: {metadata.shape}."
                )
                torch.distributed.isend(
                    tensor=metadata,
                    dst=rank,
                    group=self.group,
                )
        self.send_forward_initialized = True
        return metadata

    def init_send_backward(self, tensors: List[Optional[torch.Tensor]]):
        if self.send_backward_initialized:
            return
        logger.debug(f"Begins initializing send backward for stage {self.stage_idx}")
        if len(self.pp_backward_send_prev_infos) != 0:
            shard_info = self.pp_backward_send_prev_infos[0][1]
            assert set(
                info.num_shards for _, info in self.pp_backward_send_prev_infos
            ) == {
                shard_info.num_shards
            }, f"All tensors must have the same shard info, got {self.pp_backward_send_prev_infos}"
            metadata = self._encode_tensor_metadata(tensors)
            metadata = self._update_tensor_metadata(metadata, shard_info)
            for rank, _ in self.pp_backward_send_prev_infos:
                logger.debug(
                    f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
                    f"Sending tensor metadata to rank {rank} for stage {self.stage_idx}, "
                    f"metadata shape: {metadata.shape}."
                )
                torch.distributed.isend(
                    tensor=metadata,
                    dst=rank,
                    group=self.group,
                )
        self.send_backward_initialized = True
        return metadata

    def init_recv_forward(self):
        if self.recv_forward_tensor_properties is not None:
            return
        if len(self.pp_forward_recv_prev_infos) == 0:
            self.recv_forward_tensor_properties = []
            return
        logger.debug(f"Begins initializing recv forward for stage {self.stage_idx}")
        metadata_list = []
        reqs = []
        for rank, shard_info in self.pp_forward_recv_prev_infos:
            metadata = torch.empty(
                (self.length_of_encoded_tensor_metadata,),
                dtype=self.TENSOR_METADATA_DTYPE,
                device=torch.cuda.current_device(),
            )
            metadata_list.append(metadata)
            logger.debug(
                f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
                f"Receiving tensor metadata from rank {rank} for stage {self.stage_idx}, "
                f"expected metadata shape: {metadata.shape}."
            )
            req = torch.distributed.irecv(
                tensor=metadata,
                src=rank,
                group=self.group,
            )
            reqs.append(req)
        for req in reqs:
            req.wait()
        assert all(
            torch.allclose(metadata_list[0], metadata) for metadata in metadata_list
        ), "All metadata from different ranks should be the same"
        metadata = metadata_list[0]
        tensor_properties = self._decode_tensor_metadata(metadata)
        logger.debug(
            f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
            f"Received tensor metadata from rank {rank} for stage {self.stage_idx}. "
            f"Tensor properties: {tensor_properties}"
        )
        self.recv_forward_tensor_properties = tensor_properties

    def init_recv_backward(self):
        if self.recv_backward_tensor_properties is not None:
            return
        if len(self.pp_backward_recv_next_infos) == 0:
            self.recv_backward_tensor_properties = []
            return
        logger.debug(f"Begins initializing recv backward for stage {self.stage_idx}")
        metadata_list = []
        reqs = []
        for rank, shard_info in self.pp_backward_recv_next_infos:
            metadata = torch.empty(
                (self.length_of_encoded_tensor_metadata,),
                dtype=self.TENSOR_METADATA_DTYPE,
                device=torch.cuda.current_device(),
            )
            metadata_list.append(metadata)
            logger.debug(
                f"[PP Stage: {parallel_state.get_pipeline_parallel_stage_idx()}] "
                f"Receiving tensor metadata from rank {rank} for stage {self.stage_idx}, "
                f"expected metadata shape: {metadata.shape}."
            )
            req = torch.distributed.irecv(
                tensor=metadata,
                src=rank,
                group=self.group,
            )
            reqs.append(req)
        for req in reqs:
            req.wait()
        assert all(
            torch.allclose(metadata_list[0], metadata) for metadata in metadata_list
        ), "All metadata from different ranks should be the same"
        metadata = metadata_list[0]
        tensor_properties = self._decode_tensor_metadata(metadata)
        logger.debug(
            f"Received tensor metadata from rank {rank} for stage {self.stage_idx}. "
            f"Tensor properties: {tensor_properties}"
        )
        self.recv_backward_tensor_properties = tensor_properties

    @property
    def length_of_encoded_tensor_metadata(self) -> int:
        """Return the length of the encoded tensor metadata."""
        return 1 + self.MAX_TENSORS_TO_COMM * (self.MAX_TENSOR_NDIM + 2)

    def _encode_tensor_metadata(
        self, tensors: List[Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Encode tensors metadata into a torch Tensor for sending.

        Tensors are encoded as follows:
        - first element is the number of tensors (including the None tensors)
        - for each tensor
            - the first element is the dtype information
            - the second element is the requires_grad information
            - beginning from the second element is the shape information
                (-1 as the placeholder for non dim)
        """
        num_tensors = len(tensors)
        if num_tensors > self.MAX_TENSORS_TO_COMM:
            raise ValueError(
                f"Number of tensors to communicate ({num_tensors}) exceeds the maximum ({self.MAX_TENSORS_TO_COMM})"
            )
        tensor_metadata = torch.full(
            (self.length_of_encoded_tensor_metadata,),
            fill_value=-1,
            dtype=self.TENSOR_METADATA_DTYPE,
            device=torch.cuda.current_device(),
        )
        tensor_metadata[0] = num_tensors
        for i, tensor in enumerate(tensors):
            if tensor is None:
                continue
            dtype_int = DTYPE_TO_INT[tensor.dtype]
            requires_grad = int(tensor.requires_grad)
            start = i * (self.MAX_TENSOR_NDIM + 2) + 1
            tensor_metadata[start] = dtype_int
            tensor_metadata[start + 1] = requires_grad
            for j, dim in enumerate(tensor.shape):
                tensor_metadata[start + 2 + j] = dim

        return tensor_metadata

    def _decode_tensor_metadata(
        self, tensor_metadata: torch.Tensor
    ) -> List[Optional[Tuple[Tuple[int], torch.dtype]]]:
        """Decode tensor metadata from a torch Tensor. See `_encode_tensor_metadata` for"""
        tensor_properties = []
        num_tensors = tensor_metadata[0].item()
        for i in range(num_tensors):
            start = i * (self.MAX_TENSOR_NDIM + 2) + 1
            end = start + self.MAX_TENSOR_NDIM + 2
            dtype_int = tensor_metadata[start].item()
            if dtype_int == -1:
                tensor_properties.append(None)
                continue
            dtype = INT_TO_DTYPE[dtype_int]
            requires_grad = bool(tensor_metadata[start + 1].item())
            shape = tuple(
                dim.item() for dim in tensor_metadata[start + 2 : end] if dim != -1
            )
            tensor_properties.append((shape, dtype, requires_grad))
        return tensor_properties

    def _update_tensor_metadata(
        self, tensor_metadata: torch.Tensor, shard_info: parallel_state.ShardInfo
    ):
        """Update the tensor metadata with the shard information."""
        num_tensors = tensor_metadata[0].item()
        for i in range(num_tensors):
            start = i * (self.MAX_TENSOR_NDIM + 2) + 1
            if tensor_metadata[start].item() == -1:
                continue
            if not shard_info.is_sharded:
                continue
            num_shards = shard_info.num_shards
            assert tensor_metadata[start + 2].item() % num_shards == 0
            tensor_metadata[start + 2] = tensor_metadata[start + 2] // num_shards
        return tensor_metadata


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    model_re_swap_manager: ModelReSwapManager,
    config: MistConfig,
    forward_only: bool = False,
    deallocate_pipeline_outputs=True,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    # Compute number of warmup microbatches.
    num_microbatches = config.strategy.gradient_accumulation_steps
    num_warmup_microbatches = (
        parallel_state.get_num_pipeline_stages()
        - parallel_state.get_pipeline_parallel_stage_idx()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Init the handler if not initialized
    if not hasattr(model, "_p2p_handler"):
        model._p2p_handler = P2PHandler(config=config)
    p2p_handler = model._p2p_handler

    # Input, output tensors only need to be saved when doing backward passes
    all_input_tensors = None
    all_output_tensors = None
    if not forward_only:
        all_input_tensors = []
        all_output_tensors = []
    forward_data_store = []

    # ===========================================================================
    # Launch async opt step for the stages except the first stage
    stream = torch.cuda.Stream()
    if not parallel_state.is_pipeline_first_stage():
        model_re_swap_manager._init_optim_states_buffers()
        cuda_buffers = model_re_swap_manager.optim_states_buffers
        # cuda_buffers=None
        for optim_handle in model_re_swap_manager.module_optimizers.values():
            if optim_handle.ready_to_step:
                with torch.cuda.stream(stream):
                    optim_handle.preprocess(stream, cuda_buffers)
                    optim_handle.step()
                    optim_handle.postprocess_copy_main_params_to_model_params(stream)
                    optim_handle.postprocess_swap_out(stream)
                    optim_handle.ready_to_step = False
        # torch.cuda.empty_cache()
        model_re_swap_manager._dealloc_optim_states_buffers()


    # ===========================================================================

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        input_tensors = recv_forward(p2p_handler)
        output_tensors = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensors,
            forward_data_store,
            config,
        )
        send_forward(output_tensors, p2p_handler)

        if not forward_only:
            all_input_tensors.append(input_tensors)
            all_output_tensors.append(output_tensors)
            deallocate_output_tensor(
                output_tensors,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
            )

        # ===========================================================================
        # Debug
        # if torch.distributed.get_rank() == 3:
        #     print(
        #         f"Rank: {torch.distributed.get_rank()}. Warmup: {i}, Peak Mem: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
        #     )
        # ===========================================================================

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensors = recv_forward(p2p_handler)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        output_tensors = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensors,
            forward_data_store,
            config,
        )

        if forward_only:
            send_forward(output_tensors, p2p_handler)

            if not last_iteration:
                input_tensors = recv_forward(p2p_handler)

        else:
            output_tensor_grads = send_forward_recv_backward(
                output_tensors, p2p_handler
            )

            # Add input_tensors and output_tensor to end of list.
            all_input_tensors.append(input_tensors)
            all_output_tensors.append(output_tensors)
            deallocate_output_tensor(
                output_tensors,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
            )

            # Pop input_tensors and output_tensor from the start of the list for
            # the backward pass.
            input_tensors = all_input_tensors.pop(0)
            output_tensors = all_output_tensors.pop(0)

            input_tensor_grads = backward_step(
                input_tensors,
                output_tensors,
                output_tensor_grads,
                config,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
            )

            if last_iteration:
                input_tensors = None
                send_backward(input_tensor_grads, p2p_handler)
            else:
                input_tensors = send_backward_recv_forward(
                    input_tensor_grads, p2p_handler
                )

            # ===========================================================================
            # Debug
            # if torch.distributed.get_rank() == 3:
            #     print(
            #         f"Rank: {torch.distributed.get_rank()}. Stable: {i}, Peak Mem: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
            #     )
            # # ===========================================================================

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            # if i == num_warmup_microbatches - 1:
            #     if config.grad_sync_func is None or rank == 0:
            #         enable_grad_sync()

            input_tensors = all_input_tensors.pop(0)
            output_tensors = all_output_tensors.pop(0)

            output_tensor_grads = recv_backward(p2p_handler)

            input_tensor_grads = backward_step(
                input_tensors,
                output_tensors,
                output_tensor_grads,
                config,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
            )
            send_backward(input_tensor_grads, p2p_handler)

            # ===========================================================================
            # Debug
            # if torch.distributed.get_rank() == 3:
            #     print(
            #         f"Rank: {torch.distributed.get_rank()}. Cooldown: {i}, Peak Mem: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
            #     )
            # # ===========================================================================

    # ===========================================================================
    # Launch grad reduce scatter for the stages except the first stage
    if not parallel_state.is_pipeline_first_stage():
        for module_manager in model_re_swap_manager.module_managers.values():
            with torch.cuda.stream(stream):
                module_manager.alloc_full_grads()
                module_manager.swap_in_grads(stream)
                module_manager.reduce_grad(stream, skip=False)
                module_manager.swap_out_grads(stream)
                module_manager.dealloc_full_grads()
    # ===========================================================================

    return forward_data_store
