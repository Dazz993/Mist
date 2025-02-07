from mist.config import ModelConfig, TrainingConfig
from mist.models.gpt_config import MistGPTConfig


def add_opt_config_to_mist_gpt_config(config: MistGPTConfig, model_config: ModelConfig):
    def apply_if_not_none(key: str):
        if getattr(model_config, key, None) is not None:
            setattr(config, key, getattr(model_config, key))

    opt_knobs = [
        "use_flash_attn",
        "fused_mlp",
        "fused_dense_sqrelu_dense",
        "fused_bias_fc",
        "fused_dropout_add_ln",
        "activation_function",
        "parallel_block",
        "tensor_parallel",
        "sequence_parallel",
        "rms_norm",
        "prenorm",
        "tie_word_embeddings",
    ]
    for knob in opt_knobs:
        apply_if_not_none(knob)

    return config
