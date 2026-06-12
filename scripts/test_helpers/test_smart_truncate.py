"""Tests for smart_truncate and adaptive_max_length."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utility import smart_truncate
from adaptive_max_length import compute_max_length


# ── smart_truncate tests ─────────────────────────────────────────────────────

def test_no_truncation_needed():
    """Sequences shorter than max_length pass through unchanged."""
    sample = {
        "input_ids": [1, 2, 3, 4, 5],
        "labels": [-100, -100, -100, 10, 11],
        "attention_mask": [1, 1, 1, 1, 1],
    }
    result = smart_truncate(sample, max_length=10)
    assert result["input_ids"] == [1, 2, 3, 4, 5]
    assert result["labels"] == [-100, -100, -100, 10, 11]


def test_truncate_preserves_completion():
    """When truncation needed, completion tokens are kept, prompt is trimmed."""
    # 10 tokens: 7 prompt (-100) + 3 completion
    sample = {
        "input_ids": list(range(10)),
        "labels": [-100] * 7 + [100, 101, 102],
        "attention_mask": [1] * 10,
    }
    result = smart_truncate(sample, max_length=5)
    # Should keep all 3 completion tokens + 2 prompt tokens
    assert len(result["input_ids"]) == 5
    assert result["labels"][-3:] == [100, 101, 102]  # completion intact
    assert result["labels"][:2] == [-100, -100]  # remaining prompt
    assert result["input_ids"] == [5, 6, 7, 8, 9]


def test_truncate_prompt_heavy_28_to_1():
    """Simulates Persian-QA: 28:1 prompt:completion ratio."""
    prompt_len = 280
    comp_len = 10
    total = prompt_len + comp_len
    sample = {
        "input_ids": list(range(total)),
        "labels": [-100] * prompt_len + list(range(1000, 1000 + comp_len)),
        "attention_mask": [1] * total,
    }
    result = smart_truncate(sample, max_length=128)
    assert len(result["input_ids"]) == 128
    # All 10 completion tokens must survive
    assert result["labels"][-comp_len:] == list(range(1000, 1000 + comp_len))
    # Rest is prompt
    assert all(l == -100 for l in result["labels"][:-comp_len])


def test_truncate_completion_exceeds_max():
    """When completion alone > max_length, keep start of completion."""
    sample = {
        "input_ids": list(range(20)),
        "labels": [-100, -100] + list(range(200, 218)),  # 2 prompt + 18 comp
        "attention_mask": [1] * 20,
    }
    result = smart_truncate(sample, max_length=10)
    assert len(result["input_ids"]) == 10
    # Should start from completion_start (idx 2)
    assert result["input_ids"] == list(range(2, 12))
    assert result["labels"][0] == 200  # first completion token


def test_truncate_all_prompt():
    """All labels are -100 (no completion) — truncate from end."""
    sample = {
        "input_ids": list(range(20)),
        "labels": [-100] * 20,
        "attention_mask": [1] * 20,
    }
    result = smart_truncate(sample, max_length=10)
    assert len(result["input_ids"]) == 10
    # comp_start = len(labels) = 20, comp_len = 0
    # prompt_budget = 10 - 0 = 10, trim_start = 20 - 10 = 10
    assert result["input_ids"] == list(range(10, 20))


def test_exact_max_length():
    """Sequence exactly at max_length — no truncation."""
    sample = {
        "input_ids": [1, 2, 3],
        "labels": [-100, 5, 6],
        "attention_mask": [1, 1, 1],
    }
    result = smart_truncate(sample, max_length=3)
    assert result["input_ids"] == [1, 2, 3]


# ── adaptive_max_length tests ────────────────────────────────────────────────

def test_max_length_goes_up():
    """When data needs > 2048 and model supports it, max_length increases."""
    dist = {"p50": 500, "p95": 2500, "p99": 3000, "max": 3500, "mean": 800}
    result = compute_max_length(dist, default=2048, packing=True, model_max_length=4096)
    assert result > 2048
    assert result <= 4096


def test_max_length_capped_at_model_max():
    """max_length never exceeds model_max_length."""
    dist = {"p50": 5000, "p95": 8000, "p99": 10000, "max": 15000, "mean": 6000}
    result = compute_max_length(dist, default=2048, packing=True, model_max_length=4096)
    assert result <= 4096


def test_max_length_goes_down():
    """Short data still reduces max_length below default."""
    dist = {"p50": 50, "p95": 100, "p99": 150, "max": 200, "mean": 60}
    result = compute_max_length(dist, default=2048, packing=True, model_max_length=4096)
    assert result < 2048


def test_max_length_no_model_max():
    """Without model_max_length, caps at default (backward compatible)."""
    dist = {"p50": 500, "p95": 2500, "p99": 3000, "max": 3500, "mean": 800}
    result = compute_max_length(dist, default=2048, packing=True)
    assert result <= 2048


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
