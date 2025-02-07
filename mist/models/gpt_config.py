import inspect

from transformers import GPT2Config


class ArgumentPlaceholder:
    pass


EMPTY_ARG_PLACEHOLDER = ArgumentPlaceholder()


class MistGPTConfig(GPT2Config):
    def __init__(
        self,
        # Common arguments
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=0.00001,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
        # Uncommon arguments
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        tie_word_embeddings=True,
        # New arguments not in the original GPT2Config
        n_head_kv=EMPTY_ARG_PLACEHOLDER,
        parallel_block=EMPTY_ARG_PLACEHOLDER,
        parallel_block_tied_norm=EMPTY_ARG_PLACEHOLDER,
        rotary_emb_fraction=EMPTY_ARG_PLACEHOLDER,
        rotary_emb_interleaved=EMPTY_ARG_PLACEHOLDER,
        rotary_emb_base=EMPTY_ARG_PLACEHOLDER,
        qkv_proj_bias=EMPTY_ARG_PLACEHOLDER,
        out_proj_bias=EMPTY_ARG_PLACEHOLDER,
        mlp_fc1_bias=EMPTY_ARG_PLACEHOLDER,
        mlp_fc2_bias=EMPTY_ARG_PLACEHOLDER,
        lm_head_bias=EMPTY_ARG_PLACEHOLDER,
        pad_token_id=EMPTY_ARG_PLACEHOLDER,
        rms_norm=EMPTY_ARG_PLACEHOLDER,
        **kwargs
    ):
        new_keys = self.new_keys()
        kwargs.update(
            {
                k: v
                for k, v in locals().items()
                if k in new_keys and v is not EMPTY_ARG_PLACEHOLDER
            }
        )

        super().__init__(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            summary_type=summary_type,
            summary_use_proj=summary_use_proj,
            summary_activation=summary_activation,
            summary_proj_to_labels=summary_proj_to_labels,
            summary_first_dropout=summary_first_dropout,
            scale_attn_weights=scale_attn_weights,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
            **kwargs
        )

    @classmethod
    def new_keys(cls):
        ori_gpt2_config_signature = inspect.signature(GPT2Config.__init__)
        mist_gpt2_config_signature = inspect.signature(cls.__init__)
        return [
            k
            for k in mist_gpt2_config_signature.parameters
            if k not in ori_gpt2_config_signature.parameters
        ]
