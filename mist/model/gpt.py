from typing import Optional, Union, List, Callable

import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ProcessGroup

from mist.config import ModelConfig
from mist.model.enums import AttnMaskType
import mist.distributed.op as dist_op
from mist.modules.losses import vocab_parallel_cross_entropy
from mist.model.language_model import TransformerLanguageModel
from mist.utils.device import get_device
from mist.utils.memory import assert_viewless_tensor
from mist.distributed.overrides import MistProcessGroup
from mist.symbols import temporarily_set_sp_eq_ne


def post_language_model_processing(
    lm_output, labels, column_parallel_linear, process_group, fp16_lm_cross_entropy
):
    # Output. Format [b s h]
    lm_logits = column_parallel_linear(lm_output)

    if labels is None:
        return lm_logits.contiguous()
    else:
        if fp16_lm_cross_entropy:
            lm_logits = lm_logits.half()

        lm_logits = lm_logits.contiguous()
        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        labels = labels.contiguous()
        labels = labels.view(-1)

        loss = vocab_parallel_cross_entropy(lm_logits, labels, process_group)
        loss = loss.mean()
        # if process_group is None:
        #     loss = F.cross_entropy(lm_logits, labels)
        # else:
        #     loss = vocab_parallel_cross_entropy(lm_logits, labels, process_group)
        #     loss = loss.mean()
        return loss


class GPTModel(nn.Module):
    """GPT-2 Language model."""

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokentypes: int = 0,
        pre_process: bool = True,
        post_process: bool = True,
        process_groups: Optional[
            Union[List[dist.ProcessGroup], dist.ProcessGroup]
        ] = None,
        pre_post_process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(GPTModel, self).__init__()
        self.model_config = model_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = model_config.fp16_lm_cross_entropy
        self.tie_word_embeddings = model_config.tie_word_embeddings
        if not isinstance(process_groups, list):
            process_groups = [process_groups] * model_config.num_hidden_layers
        self.process_groups = process_groups
        self.pre_post_process_group = pre_post_process_group or process_groups[0]
        self.device = device or get_device()
        self.dtype = dtype or model_config.params_dtype

        self.language_model = TransformerLanguageModel(
            model_config,
            pre_process=pre_process,
            post_process=post_process,
            process_groups=self.process_groups,
            pre_post_process_group=self.pre_post_process_group,
            device=self.device,
            dtype=self.dtype,
        )
        self._language_model_key = "language_model"

        self._verify_config(model_config)

    def _verify_config(self, model_config: ModelConfig):
        assert (
            model_config.add_encoder
        ), "GPT model must have an encoder (which is actually a decoder)."
        assert not model_config.add_decoder, "GPT model cannot have a decoder."
        assert not model_config.add_pooler, "GPT model cannot have a pooler."
        assert not model_config.parallel_block, "GPT model cannot have parallel blocks."
        assert (
            model_config.encoder_attn_mask_type == AttnMaskType.causal
        ), "GPT model must have causal attention mask for the encoder."

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        labels=None,
        tokentype_ids=None,
        inference_params=None,
    ):

        lm_output = self.language_model(
            input_ids,
            enc_position_ids=position_ids,
            enc_attn_mask=attention_mask,
            inference_params=inference_params,
        )

        if self.post_process:
            return post_language_model_processing(
                lm_output,
                labels,
                self.language_model.output_layer,
                self.pre_post_process_group,
                self.fp16_lm_cross_entropy,
            )
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] = (
            self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars
            )
        )
        # Save word_embeddings.
        if (
            self.post_process
            and not self.pre_process
            and not not self.tie_word_embeddings
        ):
            state_dict_[self._word_embeddings_for_head_key] = (
                self.word_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
            )
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if (
            self.post_process
            and not self.pre_process
            and not not self.tie_word_embeddings
        ):
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict
            )
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)


def gpt_inputs_provider(
    batch_size: Union[int, sp.Basic],
    seq_len: Union[int, sp.Basic],
    vocab_size: Optional[Union[int, sp.Basic]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    rand: bool = True,
):
    b = batch_size
    s = seq_len
    v = vocab_size

    if rand:
        input_ids = torch.randint(0, 1000, (b, s), device=device, dtype=torch.long)
        labels = torch.randint(0, 10, (b, s), device=device, dtype=torch.long)
    else:
        input_ids = torch.ones((b, s), device=device, dtype=torch.long)
        labels = torch.ones((b, s), device=device, dtype=torch.long)

    inputs = {
        "input_ids": input_ids,
        "labels": labels,
    }

    return inputs
