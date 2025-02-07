from abc import abstractmethod
from typing import List, Optional, Tuple, Dict

import torch


class SymbolicOpContext:
    def __init__(self, 
        op, 
        saved_tensors=(), 
        extra_inner_for_fwd: Optional[List[torch.Tensor]]=None, 
        extra_inner_for_bwd: Optional[List[torch.Tensor]]=None
    ) -> None:
        if not isinstance(saved_tensors, (tuple, list)):
            raise RuntimeError(
                f"Expected saved_tensors to be tuple or list, got {type(saved_tensors)}"
            )

        self.op = op
        self.saved_tensors = tuple(saved_tensors)
        self.extra_inner_for_fwd = extra_inner_for_fwd
        self.extra_inner_for_bwd = extra_inner_for_bwd

        self.direct_producer_node = None

    def save_for_backward(self, *args):
        if self.saved_tensors is None:
            self.saved_tensors = args
        else:
            self.saved_tensors += args

    def __repr__(self):
        return f"SymbolicOpContext(op={self.op.__name__}, saved_tensors={self.saved_tensors})"


class SymbolicOp:
    def __init__(self):
        raise RuntimeError("Cannot instantiate a SymbolicOp")

    @staticmethod
    @abstractmethod
    def apply(outputs, *args, **kwargs):
        """
        Transform the outputs of the torch op to symbolic tensors.

        Parameters
        ----------
        outputs : tuple
            Outputs of the original torch op.
        args : tuple
            Arguments of the original torch op (symbolic).
        kwargs : dict
            Keyword arguments of the original torch op (symbolic).

        Returns
        -------
        transformed_outputs
            Transformed outputs.
        """
        raise NotImplementedError
