from dataclasses import dataclass
import torch


@dataclass
class HardwareSpec:
    name: str
    torch_device_type: str
    memory_in_gb: int


rtx_3090_spec = HardwareSpec(
    name="rtx_3090",
    torch_device_type="cuda",
    memory_in_gb=24,
)

t4_spec = HardwareSpec(
    name="t4",
    torch_device_type="cuda",
    memory_in_gb=16,
)

SUPPORTED_HARDWARE = {"NVIDIA GeForce RTX 3090": rtx_3090_spec, "Tesla T4": t4_spec}


def get_hardware_spec(device=None):
    device = device or torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    if device_name not in SUPPORTED_HARDWARE:
        raise RuntimeError(f"Unsupported hardware: {device_name}")
    return SUPPORTED_HARDWARE[device_name]
