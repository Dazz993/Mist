import os
import torch
import torch.nn as nn
import torch.distributed as dist

import pytest


from mist.re_swap_manager.manager import ModelReSwapManager
from mist.re_swap_manager.flat_param import HandleShardingStrategy

# Init
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def create_block():
    return torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.Linear(1024, 1024),
        torch.nn.Linear(1024, 1024),
        torch.nn.Linear(1024, 1024),
    )


class Module(nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        self.pre_linear = nn.Linear(1024, 1024)
        self.blocks = nn.ModuleList([create_block() for _ in range(n_layers)])
        self.post_linear = nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.pre_linear(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_linear(x)
        return x


@pytest.mark.parametrize("cpu_accumu_grad", [False])
@pytest.mark.parametrize("cpu_optim_step", [False])
@pytest.mark.parametrize("overlap", [False])
@pytest.mark.parametrize(
    "sharding_strategy",
    [
        # HandleShardingStrategy.ZeRO_1,
        # HandleShardingStrategy.ZeRO_2,
        HandleShardingStrategy.ZeRO_3,
    ],
)
@pytest.mark.parametrize("weight_swap_ratio", [0.0])
@pytest.mark.parametrize("grad_swap_ratio", [0.0])
@pytest.mark.parametrize("activation_swap_ratio", [0.0])
@pytest.mark.parametrize("opt_swap_ratio", [0.0])
def test_module_re_swap_optimizer(
    cpu_accumu_grad,
    cpu_optim_step,
    overlap,
    sharding_strategy,
    weight_swap_ratio,
    grad_swap_ratio,
    activation_swap_ratio,
    opt_swap_ratio,
):
    # Model
    model = Module(n_layers=4).to(device).to(dtype=torch.float16)
    for name, module in model.named_modules():
        module.name = name
    overlapped_pairs = [
        (curr_module.name, next_module.name)
        for curr_module, next_module in zip(model.blocks[:-1], model.blocks[1:])
    ]
    overlapped_pairs.append((None, model.blocks[0].name))
    overlapped_pairs.append((model.blocks[-1].name, None))
    grad_scaler = None
    model_re_swap_manager = ModelReSwapManager(
        model=model,
        overlapped_pairs=overlapped_pairs,
        process_group=dist.group.WORLD,
        grad_scaler=grad_scaler,
        cpu_accumu_grad=cpu_accumu_grad,
        cpu_optim_step=cpu_optim_step,
        overlap=overlap,
    )

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer_kwargs = {"lr": 0.01}

    for i, sub_module in enumerate(model.blocks):
        is_first = i == 0
        is_last = i == len(model.blocks) - 1
        # Apply Redundancy Elimination (RE)
        model_re_swap_manager.init_module(
            sub_module,
            state_swap_ratio=(weight_swap_ratio, grad_swap_ratio),
            activation_swap_ratio=activation_swap_ratio,
            sharding_strategy=sharding_strategy,
            device=device,
            is_first=is_first,
            is_last=is_last,
        )
        model_re_swap_manager.init_optimizer(
            sub_module,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            swap_ratio=opt_swap_ratio,
        )
        if not is_first:
            model_re_swap_manager.state_swap_managers[sub_module.name].swap_out_weights(
                torch.cuda.current_stream()
            )
    model_re_swap_manager.init_module(
        model,
        state_swap_ratio=(0.0, 0.0),
        sharding_strategy=HandleShardingStrategy.OPT_ONLY,
        device=device,
        is_root=True,
        ignored_modules=model.blocks,
    )
    model_re_swap_manager.init_optimizer(
        model,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        swap_ratio=0.0,
        is_root=True,
    )
    # Register the hooks
    model_re_swap_manager.register_hooks()
    optimizer = model_re_swap_manager.get_optimizer()
    # Synchornize the model
    torch.cuda.synchronize()

    # Check
    pre_post_layer_numel = 2 * 1024 * 1024
    block_layer_numel = 4 * 1024 * 1024
    for name, module_optimizer in model_re_swap_manager.module_optimizers.items():
        is_root = name == model.name
        opt_states = module_optimizer.opt_states
        assert len(opt_states) == 3
        for tensor in opt_states:
            if is_root:
                assert tensor.numel() == pre_post_layer_numel // world_size
            else:
                assert tensor.numel() == block_layer_numel // world_size

    # Input
    input = (
        torch.randn(1024, 1024, requires_grad=True).to(device).to(dtype=torch.float16)
    )
    # Forward
    output = model(input)
    # Backward
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()

    # Check
    flat_param_handles = model_re_swap_manager.flat_param_handles
    for (
        name,
        flat_param_group_handle,
    ) in flat_param_handles.items():
        for handle in flat_param_group_handle.handles():
            flat_param = handle.flat_param
            saved_grad_shard = flat_param._saved_grad
            if is_root:
                assert saved_grad_shard.numel() == pre_post_layer_numel
            else:
                assert saved_grad_shard.numel() == block_layer_numel // world_size

    # Step
    optimizer.step()
