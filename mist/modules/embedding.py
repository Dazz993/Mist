# Copyright (c) 2022, Tri Dao.

from typing import Callable, Optional


from einops import rearrange
import torch
import torch.nn as nn
import torch.distributed as dist
from mist.config import ModelConfig

import mist.distributed.op as dist_op
from mist.model.enums import PositionEmbeddingType, AttnMaskType
from mist.utils.device import get_device


class VocabParallelEmbedding(nn.Embedding):
    traceable = False
    __constants__ = nn.Embedding.__constants__ + ["process_group"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        process_group=None,
        padding_idx=None,
        **kwargs,
    ):
        self.process_group = process_group
        if process_group is not None:
            world_size = dist.get_world_size(process_group)
            if num_embeddings % world_size != 0:
                raise ValueError(
                    f"num_embeddings ({num_embeddings}) must be divisible by "
                    f"world_size ({world_size})"
                )
            if world_size > 1 and padding_idx is not None:
                raise RuntimeError("ParallelEmbedding does not support padding_idx")
        else:
            world_size = 1
        super().__init__(
            num_embeddings=num_embeddings // world_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            **kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.process_group is None:
            return super().forward(input)
        else:
            rank = dist.get_rank(self.process_group)
            vocab_size = self.num_embeddings
            vocab_start_index, vocab_end_index = (
                rank * vocab_size,
                (rank + 1) * vocab_size,
            )
            # Create a mask of valid vocab ids (1 means it needs to be masked).
            input_ids_mask = (input < vocab_start_index) | (input >= vocab_end_index)
            input = input - vocab_start_index
            input[input_ids_mask] = 0
            embeddings = super().forward(input)
            embeddings[input_ids_mask] = 0.0
            return embeddings

class ColumnParallelEmbedding(nn.Embedding):
    def __init__(
        self, num_embeddings, embedding_dim, *args, process_group=None, **kwargs
    ):
        self.process_group = process_group
        if process_group is not None:
            world_size = dist.get_world_size(process_group)
            if embedding_dim % world_size != 0:
                raise ValueError(
                    f"embedding_dim ({embedding_dim}) must be divisible by "
                    f"world_size ({world_size})"
                )
        else:
            world_size = 1
        super().__init__(num_embeddings, embedding_dim // world_size, *args, **kwargs)


class ParallelEmbedding(nn.Module):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        : maximum size of sequence. This
          max_sequence_length                   is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        model_config: ModelConfig,
        process_group: Optional[dist.ProcessGroup] = None,
        init_method: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(ParallelEmbedding, self).__init__()

        self.model_config = model_config
        self.init_method = init_method or (lambda x: x)
        self.process_group = process_group
        self.device = device or get_device()
        # Extract model config.
        self.hidden_size = model_config.hidden_size
        self.vocab_size = model_config.vocab_size
        self.max_position_embeddings = model_config.max_position_embeddings
        self.embedding_dropout_prob = model_config.hidden_dropout
        self.position_embedding_type = model_config.position_embedding_type
        self.num_tokentypes = model_config.num_tokentypes
        self.sequence_parallel = model_config.sequence_parallel
        if self.sequence_parallel and self.process_group is None:
            raise ValueError("sequence_parallel is set but process_group is None")
        self.fp32_residual_connection = model_config.fp32_residual_connection
        self.padding_idx = model_config.padding_idx
        self.dtype = dtype or model_config.params_dtype

        # Word embeddings (parallel).
        # NOTE(zhanda): the output is still un-allreduce'd
        self.word_embeddings = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        self._word_embeddings_key = "word_embeddings"

        # Position embedding (serial).
        if self.position_embedding_type == PositionEmbeddingType.absolute:
            assert self.max_position_embeddings is not None
            assert self.max_position_embeddings > 0
            self.position_embeddings = ColumnParallelEmbedding(
                self.max_position_embeddings,
                self.hidden_size,
                process_group=self.process_group,
                device=self.device,
                dtype=self.dtype,
            )
            self._position_embeddings_key = "position_embeddings"
            # Initialize the position embeddings.
            # if args.perform_initialization: # NOTE: always initialize them if absolute?
            self.init_method(self.position_embeddings.weight)
        else:
            self.position_embeddings = None

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = "tokentype_embeddings"
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = ColumnParallelEmbedding(
                self.num_tokentypes,
                self.hidden_size,
                process_group=self.process_group,
                device=self.device,
                dtype=self.dtype,
            )
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(self.embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")
        if dist.get_rank() == 0:
            print(
                "adding embedding for {} tokentypes".format(num_tokentypes), flush=True
            )
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = ColumnParallelEmbedding(
            num_tokentypes,
            self.hidden_size,
            process_group=self.process_group,
            device=self.device,
            dtype=self.dtype,
        )
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(
        self,
        input_ids,
        position_ids=None,
        tokentype_ids=None,
        combine_batch_seqlen_dim=False,
    ):
        # Embeddings.
        batch_size, seq_len = input_ids.size()
        world_size = (
            dist.get_world_size(self.process_group)
            if self.process_group is not None
            else 1
        )

        # Word embeddings.
        embeddings = self.word_embeddings(input_ids)

        # Partition idx.
        if world_size > 1:
            partition_dim = self.hidden_size // world_size
            rank = dist.get_rank(self.process_group)
            start = rank * partition_dim
            end = (rank + 1) * partition_dim
        else:
            start = 0
            end = self.hidden_size

        # Position embeddings.
        if self.position_embedding_type == PositionEmbeddingType.absolute:
            assert self.position_embeddings is not None
            if position_ids is None:
                position_ids = torch.arange(
                    seq_len, dtype=torch.long, device=input_ids.device
                )
            position_embeddings = self.position_embeddings(position_ids)
            embeddings[..., start:end] = (
                embeddings[..., start:end] + position_embeddings
            )
        else:
            assert self.position_embeddings is None

        # Token type embeddings.
        if self.tokentype_embeddings is not None:
            if tokentype_ids is None:
                tokentype_ids = torch.zeros(
                    seq_len, dtype=torch.long, device=input_ids.device
                )
            tokentype_embeddings = self.tokentype_embeddings(tokentype_ids)
            embeddings[..., start:end] = (
                embeddings[..., start:end] + tokentype_embeddings
            )
        else:
            assert self.num_tokentypes == 0

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Combine batch and sequence length dimension.
        if combine_batch_seqlen_dim:
            embeddings = torch.reshape(embeddings, (-1, embeddings.size(-1)))

        # Reduce-scatter or all-reduce.
        if self.process_group is not None:
            if self.sequence_parallel:
                embeddings = dist_op.reduce_scatter(embeddings, self.process_group)
            else:
                embeddings = dist_op.all_reduce(embeddings, self.process_group)

        # Dropout.
        # TODO(zhanda): check the cuda_rng_state
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] = self.word_embeddings.state_dict(
            prefix=prefix, keep_vars=keep_vars
        )
        if self.position_embedding_type == PositionEmbeddingType.absolute:
            state_dict_[self._position_embeddings_key] = (
                self.position_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
            )
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] = (
                self.tokentype_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "word_embeddings" in key:
                    state_dict_[key.split("word_embeddings.")[1]] = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self.position_embedding_type == PositionEmbeddingType.absolute:
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if "position_embeddings" in key:
                        state_dict_[key.split("position_embeddings.")[1]] = state_dict[
                            key
                        ]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if "tokentype_embeddings" in key:
                        state_dict_[key.split("tokentype_embeddings.")[1]] = state_dict[
                            key
                        ]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_, strict=strict)
            else:
                print(
                    "***WARNING*** expected tokentype embeddings in the "
                    "checkpoint but could not find it",
                    flush=True,
                )


if __name__ == "__main__":
    # Test Embedding
    dist.init_process_group(backend="gloo")
    hidden_size = 1024
    vocab_size = 32000
    max_position_embeddings = 512
    embedding_dropout_prob = 0.1
    position_embedding_type = PositionEmbeddingType.absolute
    init_method = None
    num_tokentypes = 2

    embedding = ParallelEmbedding(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        embedding_dropout_prob=embedding_dropout_prob,
        position_embedding_type=position_embedding_type,
        num_tokentypes=num_tokentypes,
        init_method=init_method,
        process_group=dist.group.WORLD,
        sequence_parallel=False,
        fp32_residual_connection=False,
    )

    input_ids = torch.randint(0, vocab_size, (8, 128), device="cuda")
    position_ids = torch.randint(0, max_position_embeddings, (128,), device="cuda")

    embeddings = embedding(input_ids, position_ids)
    print(embeddings.size())
