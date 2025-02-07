# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""BERT model."""

from typing import Optional, Union, List

import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial

from mist.config import ModelConfig
from mist.model.enums import AttnMaskType, PositionEmbeddingType
import mist.distributed.op as dist_op
from mist.modules.losses import vocab_parallel_cross_entropy
from mist.model.language_model import TransformerLanguageModel
from mist.utils.device import get_device
from mist.utils.memory import assert_viewless_tensor
from mist.modules.utils import get_activation, get_norm
from mist.modules.fused_dense import ColumnParallelLinear

# import torch
# import torch.distributed
# dist.all_reduce
# import torch.distributed.algorithms
# import torch.distributed._functional_collectives
# torch.distributed.algorithms.join
# torch.distributed._functional_collectives


def bert_extended_attention_mask(attention_mask):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)

    # Convert attention mask to binary:
    extended_attention_mask = extended_attention_mask < 0.5

    return extended_attention_mask


def bert_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids


class BertLMHead(nn.Module):
    """Masked LM head for Bert

    Arguments:
        model_config: TransformerConfig object
        mpu_vocab_size: model parallel size of vocabulary.
        parallel_output: whether output logits being distributed or not.
    """

    def __init__(
        self,
        model_config,
        mpu_vocab_size,
        process_group=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.model_config = model_config
        self.process_group = process_group
        self.device = device or get_device()
        self.dtype = dtype or model_config.params_dtype

        # self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.dense = ColumnParallelLinear(
            model_config.hidden_size,
            mpu_vocab_size,
            process_group=process_group,
            bias=True,
            device=device,
            dtype=dtype,
        )
        setattr(self.dense.weight, "sequence_parallel", model_config.sequence_parallel)
        setattr(self.dense.bias, "sequence_parallel", model_config.sequence_parallel)

        self.norm = get_norm(model_config, device=self.device, dtype=self.dtype)
        self.gelu = get_activation(model_config.activation_function)

    def forward(self, hidden_states, word_embeddings_weight=None):
        if word_embeddings_weight is not None:
            assert word_embeddings_weight.shape == self.dense.weight.shape
            self.dense.weight.data = word_embeddings_weight.data
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.norm(hidden_states)
        # output = hidden_states + self.bias
        output = hidden_states
        return output

    def load_state_dict(self, state_dict, strict=True):
        """Customize load."""

        # Handle renaming layernorm -> norm in component names
        state_dict_ = {}
        for key in state_dict.keys():
            newkey = key.replace("layernorm", "norm")
            state_dict_[newkey] = state_dict[key]

        super().load_state_dict(state_dict_, strict)


def post_language_model_processing(
    lm_output,
    pooled_output,
    labels,
    next_sentence_label,
    lm_head,
    binary_head,
    process_group,
    fp16_lm_cross_entropy,
):
    # Output. Format [b s h]
    lm_logits = lm_head(lm_output)

    binary_logits = None
    next_sentence_loss = 0
    if binary_head is not None:
        binary_logits = binary_head(pooled_output)
        if next_sentence_label is not None:
            _binary_digits = binary_logits.reshape(-1, binary_logits.size(-1))
            _next_sentence_label = next_sentence_label.reshape(-1)
            next_sentence_loss = F.cross_entropy(_binary_digits, _next_sentence_label)

    if labels is None:
        return lm_logits.contiguous(), binary_logits
    else:
        lm_logits = lm_logits.contiguous()
        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        labels = labels.contiguous()
        labels = labels.view(-1)

        if fp16_lm_cross_entropy:
            lm_logits = lm_logits.half()

        # Use vocab_parallel_cross_entropy anyway
        loss = vocab_parallel_cross_entropy(lm_logits, labels, process_group)
        loss = loss.mean()
        # if process_group is None:
        #     loss = F.cross_entropy(lm_logits, labels)
        # else:
        #     loss = vocab_parallel_cross_entropy(lm_logits, labels, process_group)
        #     loss = loss.mean()
        loss = loss + next_sentence_loss
        return loss, binary_logits


class BertModel(nn.Module):
    """Bert Language model."""

    def __init__(
        self,
        model_config,
        add_binary_head=True,
        pre_process=True,
        post_process=True,
        process_groups=None,
        pre_post_process_group=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self._verify_config(model_config)

        self.num_tokentypes = model_config.num_tokentypes
        self.fp16_lm_cross_entropy = model_config.fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        self.pre_process = pre_process
        self.post_process = post_process
        if not isinstance(process_groups, list):
            process_groups = [process_groups] * model_config.num_hidden_layers
        self.process_groups = process_groups
        self.pre_post_process_group = pre_post_process_group or process_groups[0]
        self.tie_word_embeddings = model_config.tie_word_embeddings
        self.device = device or get_device()
        self.dtype = dtype or model_config.params_dtype

        self.return_embeddings = False
        if self.return_embeddings:
            assert self.post_process and self.add_binary_head

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

        if self.post_process:
            self.lm_head = BertLMHead(
                model_config,
                mpu_vocab_size=self.language_model.embedding.word_embeddings.weight.size(
                    0
                ),
                process_group=self.pre_post_process_group,
                device=self.device,
                dtype=self.dtype,
            )
            self._lm_head_key = "lm_head"

            if self.tie_word_embeddings:
                assert (
                    self.lm_head.dense.weight.shape
                    == self.language_model.embedding.word_embeddings.weight.shape
                ), f"{self.lm_head.dense.weight.shape} != {self.language_model.embedding.word_embeddings.weight.shape}"
                self.lm_head.dense.weight = (
                    self.language_model.embedding.word_embeddings.weight
                )

            self.binary_head = None
            if self.add_binary_head:
                self.binary_head = torch.nn.Linear(
                    model_config.hidden_size,
                    2,
                    bias=False,
                    device=self.device,
                    dtype=self.dtype,
                )
                self._binary_head_key = "binary_head"

    def _verify_config(self, model_config: ModelConfig):
        assert model_config.add_encoder
        assert not model_config.add_decoder
        assert model_config.add_pooler
        assert model_config.encoder_attn_mask_type == AttnMaskType.padding
        assert model_config.position_embedding_type == PositionEmbeddingType.absolute

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids,
        attention_mask,
        tokentype_ids=None,
        lm_labels=None,
        next_sentence_label=None,
    ):

        # extended_attention_mask = bert_extended_attention_mask(attention_mask)
        extended_attention_mask = attention_mask
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids,
        )

        if self.post_process and self.add_binary_head:
            lm_output, pooled_output = lm_output

            # Return pooled output (e.g., when computing Bert embeddings).
            if self.return_embeddings:

                # Sum attention mask.
                embeddings = torch.transpose(lm_output, 0, 1)
                masks = torch.sum(attention_mask, dim=1)

                # Collect masked embeddings.
                output = torch.zeros(
                    size=(embeddings.shape[0], embeddings.shape[2]),
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                )
                for i, (embedding, mask) in enumerate(zip(embeddings, masks)):
                    output[i, :] = torch.mean(embedding[1 : mask - 1], dim=0)

                return output

        else:
            pooled_output = None

        if self.post_process:
            loss, binary_digits = post_language_model_processing(
                lm_output=lm_output,
                pooled_output=pooled_output,
                labels=lm_labels,
                next_sentence_label=next_sentence_label,
                lm_head=self.lm_head,
                binary_head=self.binary_head,
                process_group=self.pre_post_process_group,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            )
            return loss
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] = (
            self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars
            )
        )
        if self.post_process:
            state_dict_[self._lm_head_key] = (
                self.lm_head.state_dict_for_save_checkpoint(
                    prefix=prefix, keep_vars=keep_vars
                )
            )
        if self.post_process and self.add_binary_head:
            state_dict_[self._binary_head_key] = self.binary_head.state_dict(
                prefix=prefix, keep_vars=keep_vars
            )
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] = (
                self.word_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
            )
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict
        )
        if self.post_process:
            self.lm_head.load_state_dict(state_dict[self._lm_head_key], strict=strict)
        if self.post_process and self.add_binary_head:
            self.binary_head.load_state_dict(
                state_dict[self._binary_head_key], strict=strict
            )
        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict
            )


def bert_inputs_provider(
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
        input_ids = torch.randint(0, v, (b, s), device=device, dtype=torch.long, generator=rand)
        attention_mask = torch.ones((b, s), device=device, dtype=torch.bool)
        lm_labels = torch.randint(0, v, (b, s), device=device, dtype=torch.long, generator=rand)
        next_sentence_label = torch.zeros(0, (b,), device=device, dtype=torch.long, generator=rand)
        tokentype_ids = torch.randint(0, 0, (b, s), device=device, dtype=torch.long, generator=rand)
    else:
        input_ids = torch.ones((b, s), device=device, dtype=torch.long)
        attention_mask = torch.ones((b, s), device=device, dtype=torch.bool)
        lm_labels = torch.ones((b, s), device=device, dtype=torch.long)
        next_sentence_label = torch.zeros(b, device=device, dtype=torch.long)
        tokentype_ids = torch.zeros((b, s), device=device, dtype=torch.long)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokentype_ids": tokentype_ids,
        "lm_labels": lm_labels,
        "next_sentence_label": next_sentence_label,
    }

    return inputs
