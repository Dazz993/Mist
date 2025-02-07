

GPT_MODELS = {
    "1.3b": (2048, 24, 16, 50304),
    "2.7b": (2560, 32, 32, 50304),
    "7b": (4096, 32, 32, 50304),
    "13b": (5120, 40, 40, 50304),
    "20b": (6144, 44, 64, 50304),
    "40b": (8192, 48, 64, 50304),
}

def _construct_case(
    model_name: str,
    global_batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int,
    num_micro_batches: int,
    dp: int,
    tp: int,
    pp: int,
    ckpt: bool,
    zero: bool,
):
    model_config = (seq_length, hidden_size, num_layers, num_heads, vocab_size)
    return (
        model_name,
        global_batch_size,
        model_config,
        num_micro_batches,
        (dp, tp, pp, ckpt, zero),
        False,  # No profiling
    )

def benchmark()