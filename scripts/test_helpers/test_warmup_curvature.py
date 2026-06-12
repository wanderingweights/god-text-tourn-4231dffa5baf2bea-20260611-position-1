"""Warmup-derived curvature (this branch's replacement for static curvature).

Run: python scripts/test_helpers/test_warmup_curvature.py

Covers:
  1. _warmup_curvature_factor maps trajectories the right way: a fast smooth drop
     pushes LR up, flat pushes mildly down, rising clamps down, too-few-points is
     a no-op (1.0).
  2. lr_finder now emits NEUTRAL static curvature, so compute_lr_from_stats no
     longer depends on init_loss (it's deferred to the warmup).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lr_search import _warmup_curvature_factor, _WC_MIN, _WC_MAX, _WARMUP_STEPS  # noqa: E402
from lr_finder import compute_lr_from_stats  # noqa: E402


def test_factor_directions():
    up, d = _warmup_curvature_factor([5, 4.5, 4, 3.6, 3.3, 3.1, 3.0, 2.9])
    assert up > 1.0 and up <= _WC_MAX, d
    flat, _ = _warmup_curvature_factor([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    assert flat < 1.0, "flat trajectory => mild down-correction"
    rising, _ = _warmup_curvature_factor([3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4])
    assert rising == _WC_MIN, "rising loss clamps to the floor"
    none, dd = _warmup_curvature_factor([3.0, 2.5])
    assert none == 1.0 and dd["reason"] == "too_few_points"
    print(f"[ok] factor: up={up:.2f} flat={flat:.2f} rising={rising:.2f} (warmup_steps={_WARMUP_STEPS})")


def test_factor_monotonic_in_drop():
    # Larger sustained drop => larger factor (until clamp).
    small, _ = _warmup_curvature_factor([3.0, 2.98, 2.97, 2.96, 2.95, 2.95, 2.94, 2.93])
    big, _ = _warmup_curvature_factor([6.0, 5.0, 4.2, 3.6, 3.2, 3.0, 2.9, 2.85])
    assert big > small, f"bigger drop should give bigger factor: {big} vs {small}"
    print(f"[ok] monotone: small-drop={small:.2f} < big-drop={big:.2f}")


def _stats(init_loss: float) -> dict:
    return {
        "task_type": "instruct",
        "dataset": {"vocab_size": 151665},
        "weights": {"by_group": {"ffn_up": {"weight_rms": 0.028, "weight_norm": 1.0, "max_abs": 1.0}}},
        "training": {"init_loss": init_loss, "masked_completion_loss": init_loss, "gradient_noise_scale": 0.0},
    }


def test_lr_finder_curvature_now_neutral():
    # Static curvature is deferred to the warmup, so the estimate must NOT depend
    # on init_loss anymore (on main these differed via sqrt(ref/init)).
    hi = compute_lr_from_stats(_stats(13.31), "InstructTextTask", 1_540_000_000, 64)
    lo = compute_lr_from_stats(_stats(4.0), "InstructTextTask", 1_540_000_000, 64)
    assert math.isclose(hi, lo, rel_tol=1e-9), f"curvature should be neutral: {hi:.3e} vs {lo:.3e}"
    print(f"[ok] lr_finder: init_loss-independent now (LR={hi:.2e} for both 13.31 and 4.0)")


if __name__ == "__main__":
    test_factor_directions()
    test_factor_monotonic_in_drop()
    test_lr_finder_curvature_now_neutral()
    print("\nALL WARMUP-CURVATURE TESTS PASSED")
