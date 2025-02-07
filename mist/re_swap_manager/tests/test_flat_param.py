import os
import torch
import torch.nn as nn
import torch.distributed as dist

from mist.re_swap_manager.flat_param import FlatParamHandle, HandleShardingStrategy
from mist.logger import get_logger, update_logger_formatter_for_rank
from mist.re_swap_manager.mem_buffer_pool import (
    allocate_memory_buffer,
    allocate_ring_memory_buffer,
)

logger = get_logger()

# Init
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
process_group = dist.group.WORLD
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
update_logger_formatter_for_rank(logger, only_main_process_log=True)


class Module(nn.Module):
    def __init__(self, layers=2, hidden=1024):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.Linear(hidden, hidden, bias=False))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input


class Model(nn.Module):
    def __init__(self, num_blocks=1, num_layers=2, hidden=1024):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(Module(layers=num_layers, hidden=hidden))

    def forward(self, input):
        for module in self.blocks:
            input = module(input)
        return input


def test_flat_param_handle():
    model = Model().to(device)
    for name, module in model.named_modules():
        module.name = name
    # input = torch.randn(1, 1024, 1024).to(device)
    # output = model(input)
    module = model.blocks[0]
    logger.info(
        f"[0] Peaked memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )
    logger.info(
        f"[0] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    # 8MB = 8

    handle = FlatParamHandle(
        params=module.parameters(),
        module=module,
        device=device,
        sharding_strategy=HandleShardingStrategy.FULL_SHARD,
        process_group=process_group,
        dst_numel_in_cuda_for_partial_grads=0,
        dst_numel_in_cuda_for_partial_weights=0,
    )
    logger.info(f"[1] handle.flat_param.shape: {handle.flat_param.shape}")
    logger.info(
        f"[1] Peaked memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )
    logger.info(
        f"[1] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    # 4MB = 8 / 2

    # Allocate memory buffers (ring buffers for full and buffers for partial)
    allocate_ring_memory_buffer(
        name="full_weights",
        num_buffers=1,
        numel=handle.flat_param._unpadded_unsharded_size.numel(),
        dtype=handle.flat_param.dtype,
        track_usage=True,
    )
    allocate_ring_memory_buffer(
        name="full_grads",
        num_buffers=1,
        numel=handle.flat_param._unpadded_unsharded_size.numel(),
        dtype=handle.flat_param.dtype,
        track_usage=True,
    )
    for i, block in enumerate(model.blocks):
        allocate_memory_buffer(
            name=f"{block.name}_partial_weights",
            numel=0,
            dtype=handle.flat_param.dtype,
            track_usage=True,
        )
        allocate_memory_buffer(
            name=f"{block.name}_partial_grads",
            numel=0,
            dtype=handle.flat_param.dtype,
            track_usage=True,
        )
    logger.info(
        f"[2] Peaked memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )
    logger.info(
        f"[2] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    # 20MB = 8 / 2 + 8 (buffer for weights)  + 8 (buffer for grads)

    # Swap out
    handle.swap_out_weights()
    logger.info(
        f"[3] Peaked memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )
    logger.info(
        f"[3] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    torch.cuda.reset_peak_memory_stats()
    # 16MB = 8 (buffer for weights)  + 8 (buffer for grads)

    # Pre-forward
    # Swap-in and unshard
    handle.alloc_full_weights()
    handle.swap_in_weights()
    logger.info(
        f"[4] Peaked memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )
    logger.info(
        f"[4] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    # 16MB = 8 (buffer for weights)  + 8 (buffer for grads)
    torch.cuda.reset_peak_memory_stats()
    handle.unshard()
    logger.info(
        f"[5] Peaked memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )
    logger.info(
        f"[5] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    # 16MB = 8 (buffer for weights)  + 8 (buffer for grads)

    # Post-forward
    # Reshard and swap-out
    handle.reshard()
    logger.info(
        f"[6] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    handle.swap_out_weights()
    logger.info(
        f"[7] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    handle.dealloc_full_weights()
    logger.info(
        f"[8] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )

    # Pre-backward
    # Swap-in and unshard
    handle.alloc_full_weights()
    handle.alloc_full_grads()
    handle.swap_in_weights()
    handle.swap_in_grads()
    logger.info(
        f"[9] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    handle.unshard()
    logger.info(
        f"[10] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )

    # Post-backward
    # Reshard and swap-out
    handle.reshard()
    handle.reduce_grad(torch.cuda.current_stream())
    handle.swap_out_weights()
    handle.swap_out_grads()
    handle.dealloc_full_weights()
    handle.dealloc_full_grads()
    logger.info(
        f"[11] Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )


if __name__ == "__main__":
    test_flat_param_handle()
