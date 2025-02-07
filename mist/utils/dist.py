import torch
import torch.distributed as dist

def init_dummy_process_group():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="tcp://localhost:12345", rank=0, world_size=1
        )
    return dist.distributed_c10d._get_default_group()