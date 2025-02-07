from abc import abstractmethod
from typing import Callable, Sequence, List, Dict, Any, Optional
from dataclasses import dataclass
from copy import deepcopy

from mist.utils.initialization import init_empty_weights, init_empty_symbolic_weights


class ModelProvider:
    def __init__(
        self, model_cls: Callable, config: Any, n_layer_name: str, *args, **kwargs
    ):
        self.model_cls = model_cls
        self.config = config
        self.n_layer_name = n_layer_name
        self.args = args
        self.kwargs = kwargs

    def create_full_model(self, meta: bool = True):
        with init_empty_weights(enable=meta, include_buffers=True):
            model = self.model_cls(self.config, *self.args, **self.kwargs)
        return model

    def create_partial_model(self, n_layer: int = 1, meta: bool = True):
        config = deepcopy(self.config)
        n_layer_name = self.n_layer_name
        assert hasattr(config, n_layer_name), f"{n_layer_name} not found in config"
        setattr(config, n_layer_name, n_layer)

        with init_empty_weights(enable=meta, include_buffers=True):
            model = self.model_cls(config, *self.args, **self.kwargs)
        return model


class TensorParallelismMutator:
    @classmethod
    @abstractmethod
    def apply(cls, model, tp_size):
        pass

    @classmethod
    def check_or_init_mist_metadata(cls, layer):
        if not hasattr(layer, "_mist_metadata"):
            layer._mist_metadata = {}
        if "tp" not in layer._mist_metadata:
            layer._mist_metadata["tp"] = {}
        if "sync" not in layer._mist_metadata:
            layer._mist_metadata["sync"] = {}


if __name__ == "__main__":
    from transformers import BertForPreTraining, BertConfig

    config = BertConfig()
    model_provider = ModelProvider(BertForPreTraining, config, "num_hidden_layers")
    model = model_provider.create_partial_model()
