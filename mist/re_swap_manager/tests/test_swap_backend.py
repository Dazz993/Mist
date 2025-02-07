import torch

from mist.re_swap_manager.swap_backend import swap_, get_swapped


def test_basic_swap():
    before_creation_allocated = torch.cuda.memory_allocated()
    x = torch.rand(1024, 1024, device="cuda")
    numel = x.numel()
    shape = x.shape
    after_creation_allocated = torch.cuda.memory_allocated()

    swap_(x, "cpu")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == before_creation_allocated
    assert x.shape == shape
    assert x.device.type == "cpu"

    swap_(x, "cuda")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == after_creation_allocated
    assert x.shape == shape
    assert x.device.type == "cuda"

    swap_(x, "cpu")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == before_creation_allocated
    assert x.shape == shape
    assert x.device.type == "cpu"

    swap_(x, "cuda")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == after_creation_allocated
    assert x.shape == shape
    assert x.device.type == "cuda"

    del x
    assert torch.cuda.memory_allocated() == before_creation_allocated


def test_partial_swap():
    before_creation_allocated = torch.cuda.memory_allocated()
    x = torch.rand(1024, 1024, device="cuda")
    numel = x.numel()
    shape = x.shape
    after_creation_allocated = torch.cuda.memory_allocated()

    swap_ratio = 0.25
    dst_numel_in_cuda = int(numel * (1 - swap_ratio))
    swapped_nbytes = x.element_size() * dst_numel_in_cuda

    swap_(x, "partial", dst_numel_in_cuda)
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated - before_creation_allocated == swapped_nbytes
    assert tuple(x.shape) == (0,)

    swap_(x, "cuda")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == after_creation_allocated
    assert x.shape == shape
    assert x.device.type == "cuda"

    swap_(x, "partial", dst_numel_in_cuda)
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated - before_creation_allocated == swapped_nbytes
    assert tuple(x.shape) == (0,)

    swap_(x, "cpu")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == before_creation_allocated
    assert x.shape == shape
    assert x.device.type == "cpu"

    swap_(x, "cuda")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == after_creation_allocated
    assert x.shape == shape


def test_swap_in_saved_tensors():
    x = torch.rand(1024, 1024, device="cuda", requires_grad=True)
    x_shape = x.shape
    x_nbytes = x.element_size() * x.numel()
    x_numel = x.numel()
    linear = torch.nn.Linear(1024, 1024, bias=False, device="cuda")
    y = linear(x)

    before_swapping_allocated = torch.cuda.memory_allocated()
    swap_(x, "cpu")
    allocated_exclude_weights = after_swapping_allocated = torch.cuda.memory_allocated()
    assert before_swapping_allocated - after_swapping_allocated == x_nbytes
    assert x.device.type == "cpu"

    swap_(x, "cuda")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == before_swapping_allocated
    assert x.device.type == "cuda"

    swap_ratio = 0.25
    dst_numel_in_cuda = int(x_numel * (1 - swap_ratio))
    swapped_nbytes = x.element_size() * dst_numel_in_cuda

    swap_(x, "partial", dst_numel_in_cuda)
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated - allocated_exclude_weights == swapped_nbytes
    assert tuple(x.shape) == (0,)

    swap_(x, "cuda")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == before_swapping_allocated
    assert x.shape == x_shape
    assert x.device.type == "cuda"

    swap_(x, "partial", dst_numel_in_cuda)
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated - allocated_exclude_weights == swapped_nbytes
    assert tuple(x.shape) == (0,)

    swap_(x, "cpu")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == allocated_exclude_weights
    assert x.shape == x_shape
    assert x.device.type == "cpu"

    swap_(x, "cuda")
    swap_(x, "cuda")
    after_swapping_allocated = torch.cuda.memory_allocated()
    assert after_swapping_allocated == before_swapping_allocated
    assert x.shape == x_shape
    assert x.device.type == "cuda"

    # Do backward pass to ensure that the saved tensors are swapped as well
    y.sum().backward()


def test_get_swapped_with_shared_storage():
    x = torch.rand(3, 1024, 1024, device="cuda")
    a, b, c = x.chunk(3, dim=0)
    swapped = get_swapped((a, b, c), swapped_ratio=1.0)
    assert len(swapped) == 1
    assert swapped[0][0].shape == (3, 1024, 1024)
    assert swapped[0][0]._typed_storage()._data_ptr() == x._typed_storage()._data_ptr()
