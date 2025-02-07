from typing import Any
import torch
import torch.autograd


def move_to_without_grad_tracked(input: torch.Tensor, device: torch.device):
    assert isinstance(input, torch.Tensor)
    if input.is_meta:
        return torch.empty_like(input, device=device, requires_grad=input.requires_grad)
    else:
        return input.to(device)


class MoveToDevice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, device):
        ctx.device = input.device
        return move_to_without_grad_tracked(input, device)

    @staticmethod
    def backward(ctx, grad_output):
        device = ctx.device
        return move_to_without_grad_tracked(grad_output, device), None


def move_to_with_grad_tracked(input: torch.Tensor, device: torch.device):
    return MoveToDevice.apply(input, device)


if __name__ == "__main__":
    a = torch.rand(10, 10, device="meta", requires_grad=True)
    b = move_to_with_grad_tracked(a, torch.device("cuda"))
    grad_a = torch.autograd.grad(b.sum(), [a])[0]
