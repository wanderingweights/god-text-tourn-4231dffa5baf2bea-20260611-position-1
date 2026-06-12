"""
Learning rate finder based on model-prep baseline statistics.

Computes a principled starting LR from training dynamics, weight statistics,
and dataset characteristics. Replaces the SHA256 hash lookup tables that
mapped model names to pre-computed optimal learning rates.

The core idea: for Adam-family optimizers, the effective weight update per step
is approximately lr * grad_rms / sqrt(v_t). At initialization v_t ~ grad_rms^2,
so the relative weight change is roughly lr / weight_rms. We target a specific
relative weight change (eta_target) and correct for gradient noise, model scale,
and loss landscape curvature.

References:
- McCandlish et al. 2018: "An Empirical Model of Large-Batch Training"
  (gradient noise scale and critical batch size)
- Smith et al. 2018: "Don't Decay the Learning Rate, Increase the Batch Size"
  (LR-batch size equivalence and scaling laws)
"""

import math
import time
from typing import Optional


# Task-specific configuration for LR estimation.
# eta_target: target relative weight change per step (higher = more aggressive)
# ref_loss: reference loss for curvature normalization — must match the loss
#   metric returned by _get_init_loss() for that task type. For Instruct/Chat
#   this is masked_completion_loss (typically 4-7), for others it's init_loss.
# clamp: (min_lr, max_lr) safety bounds derived from empirical ranges
_TASK_CONFIG = {
    "InstructTextTask": {
        "eta_target": 0.003,
        "ref_loss": 5.5,
        "clamp": (1e-6, 1e-3),
    },
    "ChatTask": {
        "eta_target": 0.003,
        "ref_loss": 5.5,
        "clamp": (1e-6, 1e-3),
    },
    "DpoTask": {
        "eta_target": 0.001,
        "ref_loss": 2.5,
        # DPO is far more LR-sensitive than SFT: the loss only depends on the
        # chosen-rejected margin, so a too-hot LR collapses the policy's log-probs
        # to inflate the margin (the model degenerates while training loss falls).
        # The SFT-shaped formula was producing ~4e-5 here — ~5x the hand-tuned
        # dpo_config bucket and well into the danger zone. task_correction shrinks
        # the estimate into the DPO decade, and the clamp CEILING is tightened so
        # the finder can never emit an SFT-scale LR even on an odd weight_rms.
        "task_correction": 0.2,
        "clamp": (5e-7, 3e-5),
    },
    "GrpoTask": {
        "eta_target": 0.001,
        "ref_loss": 3.0,
        "clamp": (1e-6, 1.5e-5),
    },
    "EnvTask": {
        "eta_target": 0.002,
        "ref_loss": 3.0,
        "clamp": (1e-6, 2e-3),
    },
}

# Layer groups that contain trainable weights (excluding embeddings/norms
# which have different scaling properties).
_TRAINABLE_GROUPS = [
    "attention_qkv",
    "attention_output",
    "ffn_up",
    "ffn_down",
]

# Reference model size for scale factor normalization (1B parameters).
_REF_PARAMS = 1_000_000_000


def _aggregate_weight_rms(weights: dict) -> float:
    """Compute median weight RMS across trainable layer groups.

    Uses median instead of mean to be robust to outlier groups (e.g.
    fused QKV projections in Qwen2.5 where attention_qkv weight_rms
    can be 4-5x larger than other groups). Since Adam normalizes
    per-parameter, the global LR should target the typical weight
    scale — outlier groups won't be under-updated because Adam
    compensates, but small groups CAN be destabilized by a too-large LR.

    Args:
        weights: WeightStats dict with 'by_group' mapping group names
                 to dicts containing 'weight_rms', 'weight_norm', 'max_abs'.

    Returns:
        Median weight RMS across trainable groups, or a default if none found.
    """
    by_group = weights.get("by_group", {})
    rms_values = []
    for group in _TRAINABLE_GROUPS:
        if group in by_group:
            rms = by_group[group].get("weight_rms")
            if rms is not None and rms > 0:
                rms_values.append(rms)

    if not rms_values:
        for group_stats in by_group.values():
            rms = group_stats.get("weight_rms")
            if rms is not None and rms > 0:
                rms_values.append(rms)

    if not rms_values:
        return 0.02

    rms_values.sort()
    n = len(rms_values)
    if n % 2 == 1:
        return rms_values[n // 2]
    return (rms_values[n // 2 - 1] + rms_values[n // 2]) / 2


def _get_init_loss(training: dict, task_type: str) -> float:
    """Extract the most relevant initial loss metric for this task type.

    For InstructText/Chat, masked_completion_loss is more relevant than
    full-sequence init_loss since evaluation is on the task's test split.
    """
    if task_type in ("InstructTextTask", "ChatTask"):
        masked = training.get("masked_completion_loss")
        if masked is not None and masked > 0:
            return masked

    init_loss = training.get("init_loss")
    if init_loss is not None and init_loss > 0:
        return init_loss

    return 3.0  # safe default


def compute_lr_from_stats(
    baseline_stats: dict,
    task_type: str,
    param_count: int,
    effective_batch_size: int,
) -> float:
    """Compute a starting learning rate from model-prep baseline statistics.

    This is the core algorithm that replaces hash-based lookup tables.
    The estimate is principled but approximate — the probe-and-search loop
    in text_trainer.py will refine it.

    Args:
        baseline_stats: Dict from model-prep containing 'training', 'weights',
                        and 'dataset' sub-dicts. Structure matches BaselineStats
                        Pydantic models from core/models/model_prep_models.py.
        task_type: One of "InstructTextTask", "DpoTask", "GrpoTask", "ChatTask", "EnvTask".
        param_count: Total model parameter count.
        effective_batch_size: batch_size * gradient_accumulation_steps * gpu_count.

    Returns:
        Estimated starting learning rate.
    """
    t_start = time.perf_counter()
    cfg = _TASK_CONFIG.get(task_type, _TASK_CONFIG["InstructTextTask"])
    training = baseline_stats.get("training", {})
    weights = baseline_stats.get("weights", {})

    if task_type == "GrpoTask":
        lr = _compute_grpo_lr(param_count, cfg)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        print(
            f"[sn56][palpite] lr={lr:.2e} (GRPO) in {elapsed_ms:.1f}ms "
            f"(params={param_count/1e9:.2f}B, scale={math.sqrt(_REF_PARAMS / max(param_count, 1)):.3f})",
            flush=True,
        )
        return lr

    # --- SFT / DPO path (unchanged) ---

    # Step 1: Base LR from weight scale.
    weight_rms = _aggregate_weight_rms(weights)
    lr_base = cfg["eta_target"] * weight_rms

    # Step 2: Curvature correction — DEFERRED to run_lr_search on this branch.
    # The static term sqrt(ref_loss/init_loss) misreads augmentation-inflated
    # init_loss (layer_reinit pushes it toward ln(vocab), a cold start) as a sharp
    # landscape and shrinks LR with the sign backwards. This branch replaces it
    # wholesale with a factor measured from the warmup trajectory the search
    # already runs (see lr_search._warmup_curvature_factor), so we emit a neutral
    # curvature here and let the empirical signal re-centre the search.
    init_loss = _get_init_loss(training, task_type)  # kept for logging only
    curvature_factor = 1.0

    # Step 3: Gradient noise correction.
    gns = training.get("gradient_noise_scale", 0.0)
    if gns > 0 and effective_batch_size > 0:
        noise_factor = math.sqrt(
            effective_batch_size / (effective_batch_size + gns)
        )
    else:
        noise_factor = 1.0

    # Step 4: Model scale correction.
    if param_count > 0:
        scale_factor = math.sqrt(_REF_PARAMS / param_count)
    else:
        scale_factor = 1.0

    # Step 5: Task-specific corrections. Per-task multiplier (default 1.0); DPO
    # uses < 1 because its SFT-shaped estimate runs hot (see _TASK_CONFIG).
    task_correction = cfg.get("task_correction", 1.0)

    # Step 6: Combine all factors.
    lr = lr_base * curvature_factor * noise_factor * scale_factor * task_correction

    # Step 7: Clamp to empirical safe range.
    lo, hi = cfg["clamp"]
    lr = max(lo, min(hi, lr))

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    print(
        f"[sn56][palpite] lr={lr:.2e} computed in {elapsed_ms:.1f}ms "
        f"(base={lr_base:.2e}, curvature={curvature_factor:.3f}, "
        f"noise={noise_factor:.3f}, scale={scale_factor:.3f}, "
        f"task_corr={task_correction:.3f}, "
        f"weight_rms={weight_rms:.4f}, loss={init_loss:.3f}, ref_loss={cfg['ref_loss']:.1f})",
        flush=True,
    )

    return lr


# GRPO base LR for a 1B model. Literature consensus: DeepSeekMath uses 1e-6,
# DeepSeek-R1 uses 3e-6, TRL default is 5e-6, BaseWin uses 8e-6.
_GRPO_BASE_LR = 1e-5


def _compute_grpo_lr(param_count: int, cfg: dict) -> float:
    """GRPO LR: just base * 1/sqrt(N), clamped.

    GRPO gradients are policy gradients (sampled completions × advantages),
    not cross-entropy on fixed targets. SFT-derived signals (weight_rms,
    init_loss, gradient_noise_scale) don't apply. The only reliable scaling
    is model size.
    """
    scale = math.sqrt(_REF_PARAMS / max(param_count, 1))
    lr = _GRPO_BASE_LR * scale
    lo, hi = cfg["clamp"]
    return max(lo, min(hi, lr))


def estimate_starting_lr(
    baseline_stats: Optional[dict],
    task_type: str,
    param_count: int,
    effective_batch_size: int,
    fallback_lr: Optional[float] = None,
) -> Optional[float]:
    """Estimate starting LR from baseline stats, with graceful fallback.

    This is the main entry point called from config modules. If baseline_stats
    is available, computes LR from stats. Otherwise returns fallback_lr (which
    should be the size-bucket default from the config dict).

    Args:
        baseline_stats: Dict from model-prep, or None if unavailable.
        task_type: Task type string.
        param_count: Model parameter count.
        effective_batch_size: Total batch size across all accumulation and GPUs.
        fallback_lr: LR to return when baseline_stats is None.

    Returns:
        Estimated LR, or fallback_lr if stats unavailable.
    """
    if baseline_stats is None:
        if fallback_lr is not None:
            print(
                f"[sn56][palpite] No baseline stats, using fallback lr={fallback_lr:.2e}",
                flush=True,
            )
        return fallback_lr

    training = baseline_stats.get("training")
    weights = baseline_stats.get("weights")

    if not training or not weights:
        print(
            "[sn56][palpite] baseline_stats present but missing training/weights, "
            f"using fallback lr={fallback_lr}",
            flush=True,
        )
        return fallback_lr

    computed_lr = compute_lr_from_stats(
        baseline_stats, task_type, param_count, effective_batch_size
    )

    # Log comparison with bucket default for diagnostics
    if fallback_lr is not None and fallback_lr > 0 and computed_lr > 0:
        ratio = computed_lr / fallback_lr
        print(
            f"[sn56][palpite] computed={computed_lr:.2e} vs bucket={fallback_lr:.2e} "
            f"({ratio:.1f}x)",
            flush=True,
        )

    return computed_lr
