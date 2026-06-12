"""
Training data preprocessing: dedup + statistical outlier removal.

Removes exact-duplicate samples and statistical outliers (measured by
per-sample loss) before training begins. Operates on the tokenized
dataset in-place, before packing.

Outlier detection uses Median Absolute Deviation (MAD) instead of
standard deviation — robust to the right-skewed, heavy-tailed loss
distributions typical of language tasks.
"""

import os
import json
import time

import numpy as np
import torch


def deduplicate_samples(samples: list[dict]) -> list[dict]:
    """Remove exact-duplicate samples by hashing (input_ids, labels).

    Uses tuple hashing (faster than md5 for this scale). Hashes both
    input_ids and labels jointly so that samples with identical inputs
    but different labels are preserved.

    Always runs — hashing is O(n) and negligible cost.
    """
    seen = set()
    unique = []
    for sample in samples:
        ids = sample.get("input_ids", [])
        labels = sample.get("labels", [])
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        key = (tuple(ids), tuple(labels))
        if key not in seen:
            seen.add(key)
            unique.append(sample)

    n_removed = len(samples) - len(unique)
    if n_removed > 0:
        pct = n_removed / len(samples) * 100
        print(
            f"[data_filter] Dedup: {len(samples)} -> {len(unique)} "
            f"({n_removed} removed, {pct:.1f}%)",
            flush=True,
        )
    else:
        print("[data_filter] Dedup: no duplicates found", flush=True)

    return unique


@torch.no_grad()
def compute_sample_losses(
    model,
    samples: list[dict],
    batch_size: int = 128,
    device: str = "cuda",
) -> list[float]:
    """Compute per-sample cross-entropy loss via a single no-grad forward pass.

    Uses a large batch size (forward-only needs much less memory than
    forward+backward). Falls back to smaller batches on OOM.

    Returns list of float losses aligned with input samples. Samples with
    all labels=-100 get loss=0.0.
    """
    model.eval()
    all_losses = []
    t0 = time.perf_counter()

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]

        input_ids = torch.stack([
            s["input_ids"] if isinstance(s["input_ids"], torch.Tensor)
            else torch.tensor(s["input_ids"])
            for s in batch_samples
        ]).to(device)

        attention_mask = torch.stack([
            s["attention_mask"] if isinstance(s["attention_mask"], torch.Tensor)
            else torch.tensor(s["attention_mask"])
            for s in batch_samples
        ]).to(device)

        labels = torch.stack([
            s["labels"] if isinstance(s["labels"], torch.Tensor)
            else torch.tensor(s["labels"])
            for s in batch_samples
        ]).to(device)

        # Pass labels to model to avoid materializing full logits ourselves.
        # Then compute per-sample loss manually from the logits for finer control.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Per-sample cross-entropy (shift for causal LM)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        for i in range(len(batch_samples)):
            mask = shift_labels[i] != -100
            if mask.sum() == 0:
                all_losses.append(0.0)
                continue

            loss = torch.nn.functional.cross_entropy(
                shift_logits[i][mask], shift_labels[i][mask], reduction="mean"
            )
            all_losses.append(loss.item())

        # Free GPU memory between batches
        del input_ids, attention_mask, labels, logits, outputs
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    n_nonzero = sum(1 for l in all_losses if l > 0)
    print(
        f"[data_filter] Loss pass: {len(samples)} samples in {elapsed:.1f}s "
        f"({n_nonzero} with completion tokens)",
        flush=True,
    )

    model.train()
    return all_losses


def filter_outliers(
    samples: list[dict],
    losses: list[float],
    k: float = 3.0,
    min_samples: int = 200,
    save_dropped_path: str = None,
) -> list[dict]:
    """Keep samples within median ± k·MAD of the loss distribution.

    Uses Median Absolute Deviation for robustness to skewed, heavy-tailed
    loss distributions. k=3 corresponds roughly to 2σ for normal data.

    Drops zero-loss samples (all labels masked = no gradient signal).

    Args:
        samples: Training samples (list of dicts with input_ids, labels, etc.).
        losses: Per-sample losses from compute_sample_losses().
        k: Number of MADs from median. 3.0 is conservative.
        min_samples: Skip filtering if fewer samples than this.
        save_dropped_path: If set, save dropped samples + losses to this JSON file
                           for inspection.
    """
    # Drop zero-loss samples (no completion tokens = zero gradient contribution)
    n_zero = sum(1 for l in losses if l <= 0)
    if n_zero > 0:
        print(
            f"[data_filter] Dropping {n_zero} zero-loss samples (no completion tokens)",
            flush=True,
        )

    nonzero_pairs = [(s, l) for s, l in zip(samples, losses) if l > 0]
    if len(nonzero_pairs) < min_samples:
        print(
            f"[data_filter] Filter: skipped ({len(nonzero_pairs)} < {min_samples} non-zero samples)",
            flush=True,
        )
        return [s for s, _ in nonzero_pairs]

    nonzero_losses = np.array([l for _, l in nonzero_pairs])
    median_loss = float(np.median(nonzero_losses))
    mad = float(np.median(np.abs(nonzero_losses - median_loss)))

    if mad < 1e-8:
        print(
            f"[data_filter] Filter: MAD≈0 (all losses ~{median_loss:.4f}), skipped",
            flush=True,
        )
        return [s for s, _ in nonzero_pairs]

    lo = median_loss - k * mad
    hi = median_loss + k * mad

    kept = []
    dropped_low = []
    dropped_high = []

    for sample, loss in nonzero_pairs:
        if loss < lo:
            dropped_low.append((sample, loss))
        elif loss > hi:
            dropped_high.append((sample, loss))
        else:
            kept.append(sample)

    n_total = len(dropped_low) + len(dropped_high)
    pct = n_total / len(nonzero_pairs) * 100

    print(
        f"[data_filter] Filter: median={median_loss:.4f}, MAD={mad:.4f}, "
        f"band=[{lo:.4f}, {hi:.4f}] (k={k})",
        flush=True,
    )
    print(
        f"[data_filter] Dropped {len(dropped_low)} low-signal + {len(dropped_high)} outliers "
        f"= {n_total} ({pct:.1f}% of {len(nonzero_pairs)})",
        flush=True,
    )

    # Save dropped samples for inspection if requested
    if save_dropped_path and (dropped_low or dropped_high):
        dropped_info = {
            "median": median_loss,
            "mad": mad,
            "lo": lo,
            "hi": hi,
            "low_signal": [{"loss": l, "n_tokens": len([x for x in s.get("labels", []) if x != -100])} for s, l in dropped_low[:20]],
            "outliers": [{"loss": l, "n_tokens": len([x for x in s.get("labels", []) if x != -100])} for s, l in dropped_high[:20]],
        }
        with open(save_dropped_path, "w") as f:
            json.dump(dropped_info, f, indent=2)
        print(f"[data_filter] Dropped sample info saved to {save_dropped_path}", flush=True)

    return kept
