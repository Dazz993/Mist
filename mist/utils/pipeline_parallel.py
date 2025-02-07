from typing import Tuple


def calculate_num_warmup_and_1f1b_phases(
    stage_idx: int, num_stages: int, gradient_accumulation_steps: int
) -> Tuple[int, int]:
    """Calculate the number of warmup, 1f1b, and cooldown for 1F1B PP scheme."""
    warmup = num_stages - stage_idx - 1
    fb = gradient_accumulation_steps - warmup
    return warmup, fb
