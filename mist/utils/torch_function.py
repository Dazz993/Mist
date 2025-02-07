import torch


def call_ori_torch_function(func, *args, **kwargs):
    """Modified from torch._C._TensorBase.__torch_function__"""
    with torch._C.DisableTorchFunctionSubclass():
        ret = func(*args, **kwargs)
    return ret
