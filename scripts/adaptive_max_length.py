"""
Compute adaptive max sequence length from baseline stats.

Instead of using a fixed max_length (e.g. 2048), uses the actual sequence
length distribution from model-prep to right-size the training context window.
This reduces padding waste, frees memory for larger batch sizes, and increases
training throughput — all critical under a fixed time budget.

The key insight: if 99% of sequences are under 400 tokens, padding every sample
to 2048 wastes 80% of memory on zeros.
"""

from typing import Optional

# Minimum max_length to avoid degenerate cases.
_MIN_MAX_LENGTH = 128

# GPU-friendly alignment (memory allocation granularity).
_ALIGNMENT = 64


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    """Round up to the nearest multiple of alignment."""
    return ((value + alignment - 1) // alignment) * alignment


def compute_max_length(
    seq_length_distribution: Optional[dict],
    default: int = 2048,
    packing: bool = True,
    model_max_length: Optional[int] = None,
) -> int:
    """Compute optimal max_length from sequence length distribution.

    For non-packed training: sets max_length to p99 + buffer, since every
    sample is padded to max_length. Massive savings when p99 << default.

    For packed training: sets max_length to cover p99 (no truncation) but
    still caps below the default when the data is short, reducing memory
    per packed block and allowing larger batch sizes.

    When model_max_length is provided, max_length can go ABOVE the default
    to avoid truncating sequences the model can actually handle. This is
    critical for datasets like hendrycks-MATH where solutions reach 3035
    tokens but the model supports 4096.

    Args:
        seq_length_distribution: Dict with keys p50, p95, p99, max, mean.
        default: Fallback max_length when stats unavailable.
        packing: Whether dataset packing is enabled.
        model_max_length: Model's max_position_embeddings. When set,
                          allows max_length to exceed default up to this limit.

    Returns:
        Optimal max_length, aligned to GPU-friendly boundary.
    """
    if seq_length_distribution is None:
        return default

    p99 = seq_length_distribution.get("p99")
    p50 = seq_length_distribution.get("p50")

    if p99 is None or p99 <= 0:
        return default

    if packing:
        # With packing, multiple sequences fill one block of max_length.
        # We need at least p99 (to avoid truncation) and at least 2*p50
        # (to ensure reasonable packing density — at least 2 sequences/block).
        target = max(p99, 2 * (p50 or p99))
    else:
        # Without packing, every sample is padded to max_length.
        # p99 covers 99% of data; the 1% above gets truncated.
        target = p99

    # 10% buffer for tokenization variance between model-prep and training.
    target = int(target * 1.1)
    target = _align_up(target)

    # Upper bound: model's context window if available, otherwise default.
    ceiling = default
    if model_max_length is not None and model_max_length > default:
        ceiling = model_max_length

    # Clamp to [_MIN_MAX_LENGTH, ceiling].
    target = max(_MIN_MAX_LENGTH, min(target, ceiling))

    if target > default:
        increase_pct = (target / default - 1) * 100
        print(
            f"[adaptive_max_length] {target} (p99={p99}, p50={p50}, "
            f"packing={packing}, model_max={model_max_length}) — "
            f"{increase_pct:.0f}% LARGER than default {default} to avoid truncation",
            flush=True,
        )
    elif target < default:
        savings_pct = (1 - target / default) * 100
        print(
            f"[adaptive_max_length] {target} (p99={p99}, p50={p50}, "
            f"packing={packing}) — {savings_pct:.0f}% smaller than default {default}",
            flush=True,
        )
    else:
        print(
            f"[adaptive_max_length] {target} (no change from default {default})",
            flush=True,
        )

    return target
