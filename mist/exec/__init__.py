import os
import torch

import torch.distributed as dist

from mist import parallel_state
from mist.config import MistConfig


def initialize(config: MistConfig):
    # Init device
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # Init distributed
    dist.init_process_group(backend="nccl")

    # Init parallel state
    parallel_state.initialize_parallel(config)
