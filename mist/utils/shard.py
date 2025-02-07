import numpy as np
import torch
import torch.distributed as dist


def shard_tensor(
    tensor: torch.Tensor,
    num_shards: int,
    shard_id: int,
    dim: int = 0,
    share_storage: bool = False,
    buffer: torch.Tensor = None,
):
    """Shard a tensor into num_shards pieces and return the shard specified by shard_id."""
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must be between 0 and num_shards - 1")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("tensor must be a torch.Tensor")
    if not tensor.size()[dim] % num_shards == 0:
        raise ValueError("tensor dim {} must be divisible by num_shards".format(dim))

    if num_shards == 1:
        return tensor
    shard_size = tensor.size(dim) // num_shards
    shard = tensor.narrow(dim, shard_id * shard_size, shard_size)
    if not share_storage:
        if buffer is not None:
            assert buffer.numel() == shard.numel(), "buffer size mismatch"
            assert buffer.dtype == shard.dtype, "buffer dtype mismatch"
            assert (
                buffer.requires_grad == shard.requires_grad
            ), "buffer requires_grad mismatch"
            shard = buffer.view_as(shard).copy_(shard)
        else:
            shard = shard.clone()
    return shard


def unshard_tensor(
    shard: torch.Tensor,
    num_shards: int,
    process_group: dist.ProcessGroup,
    dim: int = 0,
    buffer: torch.Tensor = None,
    stream: torch.cuda.Stream = None,
    async_op: bool = False,
):
    """Concatenates a tensor that has been sharded with shard_tensor."""
    world_size = dist.get_world_size(process_group)
    stream = torch.cuda.current_stream() if stream is None else stream
    if num_shards != world_size:
        raise ValueError("num_shards must equal process_group.size()")
    if not isinstance(shard, torch.Tensor):
        raise ValueError("shard must be a torch.Tensor")

    full_size = list(shard.size())
    full_size[dim] *= num_shards
    if buffer is not None:
        assert buffer.numel() == np.prod(full_size), "buffer size mismatch"
        # assert buffer.dtype == shard.dtype, "buffer dtype mismatch"
        assert (
            buffer.requires_grad == shard.requires_grad
        ), "buffer requires_grad mismatch"
        full_tensor = buffer.view(full_size).data
    else:
        full_tensor = torch.empty(
            full_size,
            dtype=shard.dtype,
            device=shard.device,
            requires_grad=shard.requires_grad,
        )

    if shard.is_cpu:
        tensor_list = list(torch.chunk(full_tensor, world_size, dim=dim))
        # TODO: check the input process group is correct
        # process_group = dist.new_group(backend="gloo")
        work = dist.all_gather(
            tensor_list=tensor_list,
            tensor=shard,
            group=process_group,
            async_op=async_op,
        )
    else:
        with torch.cuda.stream(stream):
            # print(f"[RANK={torch.distributed.get_rank()}] {shard=}, {full_tensor=}")
            dist.all_gather_into_tensor(
                output_tensor=full_tensor,
                input_tensor=shard,
                group=process_group,
                async_op=async_op,
            )
    return full_tensor
