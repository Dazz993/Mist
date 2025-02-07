from abc import ABC
from abc import abstractmethod

import torch

class BaseGradScaler(ABC):

    def __init__(self, initial_scale):
        """Initialize scale value with the input initial scale."""
        assert initial_scale > 0.0
        self._scale = torch.cuda.FloatTensor([initial_scale])

        self.found_inf = torch.cuda.FloatTensor([0.0])

    @property
    def scale(self):
        return self._scale

    @property
    def inv_scale(self):
        return self._scale.double().reciprocal().float()

    def _get_scale_async(self):
        return self._scale

    def scale(self, loss):
        return loss * self._scale

    def unscale_(self, grads):
        torch._amp_foreach_non_finite_check_and_unscale_(grads, self.found_inf, self.inv_scale)

class ConstantGradScaler(BaseGradScaler):
    def __init__(self, scale):
        super().__init__(scale)
        self._inv_scale = self._scale.double().reciprocal().float() 

    @property
    def inv_scale(self):
        return self._inv_scale

    def step(self, optimizer, closure=None):
        optimizer.step(closure)