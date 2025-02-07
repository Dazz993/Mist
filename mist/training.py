import torch
import torch.distributed as dist

from mist import parallel_state
from mist.config import MistConfig
from mist.logger import get_logger
from mist.pipeline_parallel.schedules import get_forward_backward_func

logger = get_logger(__name__)


def train_step(
    forward_step_func,
    data_iterator,
    model,
    model_re_swap_manager,
    config: MistConfig,
):
    forward_backward_func = get_forward_backward_func()

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        model_re_swap_manager=model_re_swap_manager,
        config=config,
        forward_only=False,
    )

    return losses_reduced
