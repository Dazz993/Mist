import numpy as np
import torch
import torch.distributed as dist

from mist.node_database.inputs_outputs_spec import InputsSpec, TensorSpec
from mist.utils.common import benchmark_func, process_benchmarking_results

GB = 1024**3
MB = 1024**2
KB = 1024


def benchmark_raw_communication_bandwidth(shape, dtype, src_device, dst_device):
    element_size = torch.tensor(1, dtype=dtype).element_size()

    def prepare_func():
        src_tensor_spec = TensorSpec(shape, dtype, requires_grad=False)
        dst_tensor_spec = TensorSpec(shape, dtype, requires_grad=False)
        src_tensor = src_tensor_spec.instantiate(src_device, rand=True)
        dst_tensor = dst_tensor_spec.instantiate(dst_device, rand=False)
        return src_tensor, dst_tensor

    def func(src_tensor, dst_tensor):
        dst_tensor.copy_(src_tensor)

    costs = benchmark_func(
        func,
        prepare_func=prepare_func,
        warmup=5,
        number=10,
        sync_func=torch.cuda.synchronize,
        enable_tqdm=True,
    )

    size = np.prod(shape) * element_size

    process_benchmarking_results(
        costs, msg=f"{src_device} -> {dst_device} ({size})", _print=True
    )

    print(f"Bandwidth: {size / GB / np.mean(costs):.2f} GB/s")

    return


def main():
    shape = (32, 1024, 1024)
    dtype = torch.int8

    benchmark_raw_communication_bandwidth(shape, dtype, "cpu", "cpu")

    if dist_initialized:
        benchmark_raw_communication_bandwidth(
            shape, dtype, "cpu", f"cuda:{dist.get_rank()}"
        )


if __name__ == "__main__":
    main()
