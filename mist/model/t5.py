# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""T5 model."""

from typing import Optional, Union, List, Callable, Tuple

import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from mist.config import ModelConfig
from mist.model.enums import AttnMaskType, PositionEmbeddingType
import mist.distributed.op as dist_op
from mist.modules.losses import vocab_parallel_cross_entropy
from mist.model.language_model import TransformerLanguageModel
from mist.utils.device import get_device
from mist.utils.memory import assert_viewless_tensor
from mist.modules.utils import get_activation, get_norm
from mist.modules.fused_dense import ColumnParallelLinear, fused_dense_func
from mist.modules.inference_params import InferenceParams


def t5_extended_attention_mask(attention_mask_list):

    def attn_mask_postprocess(attn_mask):
        # [b, 1, s, s]
        extended_attention_mask = attn_mask.unsqueeze(1)
        return extended_attention_mask

    return [attn_mask_postprocess(attn_mask) for attn_mask in attention_mask_list]


def t5_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids


class T5LMHead(nn.Module):
    """Masked LM head for T5

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, process_group, device=None, dtype=None):
        super(T5LMHead, self).__init__()

        self.bias = torch.nn.Parameter(
            torch.zeros(mpu_vocab_size, device=device, dtype=dtype)
        )
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.process_group = process_group
        self.device = device
        self.dtype = dtype

    def forward(self, hidden_states, word_embeddings_weight):
        output = fused_dense_func(
            x=hidden_states,
            weight=word_embeddings_weight,
            bias=self.bias,
            process_group=self.process_group,
        )
        return output


class T5Model(nn.Module):
    """T5 Language model."""

    def __init__(
        self,
        model_config,
        pre_process=True,
        post_process=True,
        process_groups=None,
        pre_post_process_group=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self._verify_config(model_config)

        self.fp16_lm_cross_entropy = model_config.fp16_lm_cross_entropy
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = model_config.add_encoder
        self.add_decoder = model_config.add_decoder
        if not isinstance(process_groups, list):
            process_groups = [process_groups] * model_config.num_hidden_layers
        self.process_groups = process_groups
        self.pre_post_process_group = pre_post_process_group or process_groups[0]
        self.device = device or get_device()
        self.dtype = dtype or model_config.params_dtype
        self.tie_word_embeddings = model_config.tie_word_embeddings

        self.language_model = TransformerLanguageModel(
            model_config=model_config,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pre_post_process_group=self.pre_post_process_group,
            process_groups=self.process_groups,
            device=self.device,
            dtype=self.dtype,
        )
        self._language_model_key = "language_model"

        if self.post_process and self.add_decoder:
            self.lm_head = T5LMHead(
                self.language_model.embedding.word_embeddings.weight.size(0),
                self.pre_post_process_group,
                device=self.device,
                dtype=self.dtype,
            )
            self._lm_head_key = "lm_head"
            if not self.tie_word_embeddings:
                self.lm_head_weight = torch.nn.Parameter(
                    torch.empty_like(
                        self.language_model.embedding.word_embeddings.weight
                    )
                )
            else:
                self.lm_head_weight = self.language_model.embedding.word_embeddings

    def _verify_config(self, model_config: ModelConfig):
        assert model_config.add_encoder
        assert model_config.add_decoder
        assert not model_config.add_pooler
        assert model_config.encoder_attn_mask_type == AttnMaskType.padding
        assert model_config.position_embedding_type == PositionEmbeddingType.absolute
        assert model_config.decoder_attn_mask_type == AttnMaskType.causal

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attn_mask: torch.Tensor,
        decoder_attn_mask: torch.Tensor,
        encoder_decoder_attn_mask: torch.Tensor,
        tokentype_ids: Optional[torch.Tensor] = None,
        lm_labels: torch.Tensor = None,
        enc_hidden_states: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
    ):
        """Forward pass.

        Args:
            encoder_input_ids (Tensor): input ids for encoder
            decoder_input_ids (Tensor): input ids for decoder
            encoder_attn_mask (Tensor): self-attention mask for encoder
            decoder_attn_mask (Tensor): self-attention mask for decoder
            encoder_decoder_attn_mask (Tensor): cross-attention mask between encoder and decoder
            lm_labels (Tensor): labels for decoder output
            inference_params (InferenceParams): relevant arguments for inferencing

        Returns:
            Tensor: loss tensor
        """

        # Converting the attention masks to proper parameter settings
        # encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask = (
        #     t5_extended_attention_mask(
        #         [encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask]
        #     )
        # )

        encoder_position_ids = t5_position_ids(encoder_input_ids)
        decoder_position_ids = t5_position_ids(decoder_input_ids)

        lm_output = self.language_model(
            enc_input_ids=encoder_input_ids,
            enc_position_ids=encoder_position_ids,
            enc_attn_mask=encoder_attn_mask,
            dec_input_ids=decoder_input_ids,
            dec_position_ids=decoder_position_ids,
            dec_attn_mask=decoder_attn_mask,
            enc_dec_attn_mask=encoder_decoder_attn_mask,
            tokentype_ids=tokentype_ids,
            enc_hidden_states=enc_hidden_states,
        )

        if self.post_process and self.add_decoder:
            decoder_output, encoder_output = lm_output
            lm_logits = self.lm_head(decoder_output, self.lm_head_weight)

            if lm_labels is None:
                return lm_logits.contiguous()
            else:
                lm_logits = lm_logits.contiguous()
                lm_logits = lm_logits.view(-1, lm_logits.size(-1))
                lm_labels = lm_labels.contiguous()
                lm_labels = lm_labels.view(-1)
                if self.fp16_lm_cross_entropy:
                    lm_logits = lm_logits.half()

                # if self.pre_post_process_group is None:
                #     loss = F.cross_entropy(lm_logits, lm_labels)
                # else:
                #     loss = vocab_parallel_cross_entropy(
                #         lm_logits, lm_labels, self.pre_post_process_group
                #     )
                #     loss = loss.mean()
                loss = vocab_parallel_cross_entropy(
                    lm_logits, lm_labels, self.pre_post_process_group
                )
                loss = loss.mean()
                return loss

        elif self.add_decoder and not self.add_encoder:
            decoder_output, encoder_output = lm_output
            return decoder_output

        else:
            encoder_output = lm_output
            return encoder_output

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] = (
            self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars
            )
        )
        if self.post_process and self.add_decoder:
            state_dict_[self._lm_head_key] = (
                self.lm_head.state_dict_for_save_checkpoint(
                    prefix=prefix, keep_vars=keep_vars
                )
            )
        # Save word_embeddings.
        if self.post_process and not self.pre_process and self.add_decoder:
            state_dict_[self._word_embeddings_for_head_key] = (
                self.word_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
            )
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict
        )
        if self.post_process and self.add_decoder:
            self.lm_head.load_state_dict(state_dict[self._lm_head_key], strict=strict)
        # Load word embeddings.
        if self.post_process and not self.pre_process and self.add_decoder:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict
            )


def t5_inputs_provider(
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
        encoder_input_ids = torch.randint(v, (b, s), device=device, dtype=torch.long)
        decoder_input_ids = torch.randint(v, (b, s), device=device, dtype=torch.long)
        encoder_attn_mask = torch.ones((b, s), device=device, dtype=bool)
        decoder_attn_mask = torch.ones((b, s), device=device, dtype=bool)
        encoder_decoder_attn_mask = torch.ones((b, s), device=device, dtype=bool)
        lm_labels = torch.randint(v, (b, s), device=device, dtype=torch.long)
    else:
        encoder_input_ids = torch.ones((b, s), device=device, dtype=torch.long)
        decoder_input_ids = torch.ones((b, s), device=device, dtype=torch.long)
        encoder_attn_mask = torch.ones((b, s), device=device, dtype=bool)
        decoder_attn_mask = torch.ones((b, s), device=device, dtype=bool)
        encoder_decoder_attn_mask = torch.ones((b, s), device=device, dtype=bool)
        lm_labels = torch.ones((b, s), device=device, dtype=torch.long)

    inputs = {
        "encoder_input_ids": encoder_input_ids,
        "decoder_input_ids": decoder_input_ids,
        "encoder_attn_mask": encoder_attn_mask,
        "decoder_attn_mask": decoder_attn_mask,
        "encoder_decoder_attn_mask": encoder_decoder_attn_mask,
        "lm_labels": lm_labels,
    }

    return inputs
