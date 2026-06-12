"""Tests for naive packing, Gaussian sampling, prompt loss weight, and coverage estimation."""
import sys
import os
import types

# Stub heavy deps
_tf = types.ModuleType('transformers')
_tf.AutoTokenizer = type('AutoTokenizer', (), {})
_tf_tu = types.ModuleType('transformers.trainer_utils')
_tf_tu.is_main_process = lambda x: True
sys.modules['wandb'] = types.ModuleType('wandb')
sys.modules['transformers'] = _tf
sys.modules['transformers.trainer_utils'] = _tf_tu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utility import gaussian_subsample, apply_prompt_loss_weight, smart_truncate


# ── Naive packing tests ──────────────────────────────────────────────────────

def test_naive_packing_basic():
    """Naive packing concatenates sequences with position_ids reset."""
    # Import after stubs
    from monkeypatch import pack_data_points_naive

    class FakeTokenizer:
        pad_token_id = 0
        padding_side = "right"

    items = [
        {"input_ids": [1, 2, 3], "labels": [-100, 10, 11], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5], "labels": [-100, 20], "attention_mask": [1, 1]},
    ]
    result = pack_data_points_naive(items, FakeTokenizer(), model_max_length=8)

    # Check concatenation + padding
    assert result["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, 0, 0]
    # First token of each seq has label=-100
    assert result["labels"].tolist() == [-100, 10, 11, -100, 20, -100, -100, -100]
    # Position IDs reset per sequence
    assert result["position_ids"].tolist() == [0, 1, 2, 0, 1, 0, 0, 0]
    # Standard 1D attention mask
    assert result["attention_mask"].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    print("PASS naive_packing_basic")


def test_naive_packing_single_item():
    """Single item packing should work."""
    from monkeypatch import pack_data_points_naive

    class FakeTokenizer:
        pad_token_id = 0
        padding_side = "right"

    items = [{"input_ids": [1, 2], "labels": [-100, 5], "attention_mask": [1, 1]}]
    result = pack_data_points_naive(items, FakeTokenizer(), model_max_length=4)
    assert result["input_ids"].tolist() == [1, 2, 0, 0]
    assert result["position_ids"].tolist() == [0, 1, 0, 0]
    print("PASS naive_packing_single_item")


def test_packed_dataset_use_fa_flag():
    """PackedDataset respects use_fa flag."""
    from monkeypatch import PackedDataset, pack_data_points_FA, pack_data_points_naive

    class FakeTokenizer:
        pad_token_id = 0
        padding_side = "right"

    class FakeDataset:
        eval_dataset = [
            {"input_ids": [1, 2, 3], "labels": [-100, 10, 11], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5], "labels": [-100, 20], "attention_mask": [1, 1]},
        ]

    ds_naive = PackedDataset(FakeDataset(), FakeTokenizer(), max_input_length=8, use_fa=False)
    item = ds_naive[0]
    # Naive packing should have position_ids
    assert "position_ids" in item, "Naive packing should include position_ids"

    ds_fa = PackedDataset(FakeDataset(), FakeTokenizer(), max_input_length=8, use_fa=True)
    item_fa = ds_fa[0]
    # FA packing should NOT have position_ids
    assert "position_ids" not in item_fa, "FA packing should not include position_ids"
    print("PASS packed_dataset_use_fa_flag")


# ── Gaussian sampling tests ──────────────────────────────────────────────────

def test_gaussian_subsample_reduces_size():
    """Gaussian subsampling returns exactly target_size samples."""
    samples = [{"labels": [-100] * 5 + [i] * 10} for i in range(1000)]
    result = gaussian_subsample(samples, target_size=500)
    assert len(result) == 500
    print("PASS gaussian_subsample_reduces_size")


def test_gaussian_subsample_prefers_middle():
    """Gaussian sampling should oversample medium-difficulty, undersample extremes."""
    import numpy as np

    # Create dataset with clear difficulty tiers
    easy = [{"labels": [-100] * 10 + [1] * 5} for _ in range(200)]       # 5 comp tokens
    medium = [{"labels": [-100] * 10 + [1] * 50} for _ in range(200)]    # 50 comp tokens
    hard = [{"labels": [-100] * 10 + [1] * 200} for _ in range(200)]     # 200 comp tokens
    samples = easy + medium + hard  # 600 total

    result = gaussian_subsample(samples, target_size=300, seed=42)

    # Count how many from each tier survived
    easy_count = sum(1 for s in result if sum(1 for l in s["labels"] if l != -100) == 5)
    medium_count = sum(1 for s in result if sum(1 for l in s["labels"] if l != -100) == 50)
    hard_count = sum(1 for s in result if sum(1 for l in s["labels"] if l != -100) == 200)

    # Medium should have the most
    assert medium_count > easy_count, f"Medium {medium_count} should > easy {easy_count}"
    assert medium_count > hard_count, f"Medium {medium_count} should > hard {hard_count}"
    print(f"PASS gaussian_prefers_middle (easy={easy_count}, med={medium_count}, hard={hard_count})")


def test_gaussian_subsample_no_op_when_small():
    """If target_size >= dataset size, return all samples."""
    samples = [{"labels": [1, 2, 3]} for _ in range(100)]
    result = gaussian_subsample(samples, target_size=200)
    assert len(result) == 100
    print("PASS gaussian_subsample_no_op")


# ── Prompt loss weight tests ─────────────────────────────────────────────────

def test_plw_unmasks_some_prompt_tokens():
    """PLW should unmask ~5% of prompt tokens."""
    samples = [
        {"input_ids": list(range(100)), "labels": [-100] * 90 + list(range(100, 110))}
        for _ in range(100)  # 100 independent copies
    ]

    result = apply_prompt_loss_weight(samples, plw=0.05, seed=42)

    # Count unmasked prompt tokens
    total_prompt = 0
    unmasked = 0
    for s in result:
        for i, (tok, lab) in enumerate(zip(s["input_ids"], s["labels"])):
            if i < 90:  # prompt region
                total_prompt += 1
                if lab != -100:
                    unmasked += 1

    pct = unmasked / total_prompt
    assert 0.02 < pct < 0.10, f"Expected ~5% unmasked, got {pct:.1%}"
    print(f"PASS plw_unmasks_prompt ({pct:.1%} unmasked)")


def test_plw_preserves_completion():
    """PLW should not change completion tokens."""
    import copy
    samples = [{"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, -100, 10, 11]}]
    samples = copy.deepcopy(samples)
    result = apply_prompt_loss_weight(samples, plw=0.5, seed=42)

    # Completion tokens (last 2) should be unchanged
    assert result[0]["labels"][3] == 10
    assert result[0]["labels"][4] == 11
    print("PASS plw_preserves_completion")


def test_plw_zero_weight_no_change():
    """PLW=0 should not unmask anything."""
    import copy
    samples = [{"input_ids": [1, 2, 3], "labels": [-100, -100, 5]}]
    samples = copy.deepcopy(samples)
    result = apply_prompt_loss_weight(samples, plw=0.0, seed=42)
    assert result[0]["labels"] == [-100, -100, 5]
    print("PASS plw_zero_no_change")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
