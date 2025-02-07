from typing import Dict
import torch
from torch import nn, fx
from copy import deepcopy

from mist.utils.initialization import init_empty_weights
from mist.logger import get_logger

from transformers import AutoConfig

logger = get_logger()

# Will be created at the end of this file
SUPPORTED_MODEL_NAMES = None


class HFProvider:
    MODEL_TYPE: str
    DEFAULT_MODEL_NAME: str
    STANDARD2CUSTOM = {
        "hidden_size": "hidden_size",
        "n_layers": "n_layers",
        "n_heads": "n_heads",
        "vocab_size": "vocab_size",
    }
    TINY_CONFIGS = {
        "hidden_size": 256,
        "n_layers": 1,
        "n_heads": 32,
    }

    @classmethod
    def supported_model_types(cls):
        """
        Return a dictionary mapping model types to model provider classes
        """
        subclasses = cls.__subclasses__()
        model_types2classes = {subclass.MODEL_TYPE: subclass for subclass in subclasses}
        return model_types2classes

    def get_config(self, model_name=None):
        """
        Get the model config from the model name. Should be overriden by subclasses if necessary.
        """
        model_name = model_name or self.DEFAULT_MODEL_NAME
        return AutoConfig.from_pretrained(model_name)

    def get_model(self, config, device="meta", **kwargs):
        """
        Get the model from the config. By default, the subclass should implement _get_model.
        Could be overriden by subclasses if necessary.
        """
        config = config or self.get_config()
        with init_empty_weights(enable=(device == "meta"), include_buffers=True):
            model = self._get_model(config, **kwargs)
        return model

    def get_tiny_config(self, model_name=None):
        config = self.get_config(model_name)
        for key, value in self.TINY_CONFIGS.items():
            real_key = self.STANDARD2CUSTOM[key]
            setattr(config, real_key, value)
        return config

    @classmethod
    def get_tiny_model(cls, device="meta", **kwargs):
        model_provider = cls()
        config = model_provider.get_tiny_config()
        model = model_provider.get_model(config, device=device, **kwargs)
        return model, config

    def _get_model(self, config, **kwargs):
        raise NotImplementedError(
            f"Model provider {self.__class__.__name__} does not implement _get_model"
        )

    @classmethod
    def from_type_or_name(
        cls, model_type=None, model_name=None, device="meta", **kwargs
    ):
        if cls is HFProvider:
            # Check whether the model type is supported
            # if supported, get the corresponding model provider
            supported_model_types: Dict[str, HFProvider] = cls.supported_model_types()
            assert (
                model_type is not None
            ), "model_type must be specified if cls is HFProvider"
            if model_type not in supported_model_types:
                raise ValueError(
                    f"Model type {model_type} not supported. Supported model types: {list(supported_model_types.keys())}"
                )
            model_provider_cls = supported_model_types[model_type]
            model_provider = model_provider_cls()

        else:
            model_provider_cls = cls
            model_provider = cls()

        model_name = model_name or model_provider_cls.DEFAULT_MODEL_NAME

        # Get the model, model config
        config = model_provider.get_config(model_name)
        model = model_provider.get_model(config, device=device, **kwargs)
        return model, config


class BertProvider(HFProvider):
    MODEL_TYPE = "bert"
    DEFAULT_MODEL_NAME = "bert-base-uncased"
    STANDARD2CUSTOM = {
        "hidden_size": "hidden_size",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "vocab_size": "vocab_size",
    }

    def _get_model(self, config, **kwargs):
        from transformers import BertForPreTraining

        model = BertForPreTraining(config, **kwargs)
        return model


class RobertaProvider(HFProvider):
    MODEL_TYPE = "roberta"
    DEFAULT_MODEL_NAME = "roberta-base"
    STANDARD2CUSTOM = {
        "hidden_size": "hidden_size",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "vocab_size": "vocab_size",
    }

    def _get_model(self, config, **kwargs):
        from transformers import RobertaForCausalLM

        model = RobertaForCausalLM(config, **kwargs)
        return model


class GPT2Provider(HFProvider):
    MODEL_TYPE = "gpt2"
    DEFAULT_MODEL_NAME = "gpt2"
    STANDARD2CUSTOM = {
        "hidden_size": "n_embd",
        "n_layers": "n_layer",
        "n_heads": "n_head",
        "vocab_size": "vocab_size",
    }

    def _get_model(self, config, **kwargs):
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel(config, **kwargs)
        return model


class GPTJProvider(HFProvider):
    MODEL_TYPE = "gpt-j"
    DEFAULT_MODEL_NAME = "EleutherAI/gpt-j-6B"
    STANDARD2CUSTOM = {
        "hidden_size": "n_embd",
        "n_layers": "n_layer",
        "n_heads": "n_head",
        "vocab_size": "vocab_size",
    }

    def _get_model(self, config, **kwargs):
        from transformers import GPTJForCausalLM

        model = GPTJForCausalLM(config, **kwargs)
        return model


class GPTNeoProvider(HFProvider):
    MODEL_TYPE = "gpt-neo"
    DEFAULT_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
    STANDARD2CUSTOM = {
        "hidden_size": "hidden_size",
        "n_layers": "num_layers",
        "n_heads": "num_heads",
        "vocab_size": "vocab_size",
    }

    def _get_model(self, config, **kwargs):
        from transformers import GPTNeoForCausalLM

        model = GPTNeoForCausalLM(config, **kwargs)
        return model


class OPTProvider(HFProvider):
    MODEL_TYPE = "opt"
    DEFAULT_MODEL_NAME = "facebook/opt-350m"
    STANDARD2CUSTOM = {
        "hidden_size": "hidden_size",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "vocab_size": "vocab_size",
    }

    def _get_model(self, config, **kwargs):
        from transformers import OPTModel

        model = OPTModel(config, **kwargs)
        return model


class LLaMAProvider(HFProvider):
    MODEL_TYPE = "llama"
    DEFAULT_MODEL_NAME = "dummy-7b-styled"
    STANDARD2CUSTOM = {
        "hidden_size": "hidden_size",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "vocab_size": "vocab_size",
    }

    def get_config(self, model_name=None):
        from transformers import LlamaConfig

        logger.warning(f"Using dummy config for LLaMA")
        return LlamaConfig()

    def _get_model(self, config, **kwargs):
        from transformers import LlamaModel

        model = LlamaModel(config, **kwargs)
        return model


class T5Provider(HFProvider):
    MODEL_TYPE = "t5"
    DEFAULT_MODEL_NAME = "t5-small"
    STANDARD2CUSTOM = {
        "hidden_size": "d_model",
        "n_layers": "num_layers",
        "n_heads": "num_heads",
        "vocab_size": "vocab_size",
    }

    def _get_model(self, config, **kwargs):
        from transformers import T5ForConditionalGeneration

        model = T5ForConditionalGeneration(config, **kwargs)
        return model


SUPPORTED_MODEL_NAMES = list(HFProvider.supported_model_types().keys())

if __name__ == "__main__":
    BertProvider.from_type_or_name()
    RobertaProvider.from_type_or_name()
    GPT2Provider.from_type_or_name()
    GPTJProvider.from_type_or_name()
    OPTProvider.from_type_or_name()
    LLaMAProvider.from_type_or_name()
    T5Provider.from_type_or_name()
