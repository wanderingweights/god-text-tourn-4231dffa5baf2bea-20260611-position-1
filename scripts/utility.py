from typing import Dict, Any
import yaml
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from typing import Callable
import torch
import logging
from datetime import datetime
import sys
import wandb
import random
import json
import requests
import os
import shutil
from transformers.trainer_utils import is_main_process

logger = logging.getLogger()
logger.setLevel(logging.INFO)
 # Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
 # Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))



def log_info(message: str, event_name: str = "print"):
    if is_main_process(LOCAL_RANK):
        logger.info(f"{event_name}: {message}")
    # wandb.log({"event": event_name, "message": message})


def pad_sequence(sequence: list[int], pad_value: int, max_length: int, padding_side: str) -> list[int]:
    if padding_side == "left":
        return [pad_value] * (max_length - len(sequence)) + sequence
    else:
        return sequence + [pad_value] * (max_length - len(sequence))


def smart_truncate(input_dict: dict, max_length: int) -> dict:
    """Truncate from the PROMPT side to preserve completion tokens.

    Standard truncation (from the right) chops off the end of completions —
    for math/reasoning tasks that's the answer itself. Instead, find where
    labels switch from -100 (prompt) to real values (completion) and trim
    from the prompt side.
    """
    input_ids = input_dict["input_ids"]
    if len(input_ids) <= max_length:
        return input_dict

    labels = input_dict["labels"]

    # Find completion start: first non -100 label
    comp_start = len(labels)  # default: all prompt
    for i, l in enumerate(labels):
        if l != -100:
            comp_start = i
            break

    comp_len = len(input_ids) - comp_start

    if comp_len >= max_length:
        # Completion alone exceeds max_length — keep start of completion
        # (still better than keeping prompt + truncated completion)
        trim_start = comp_start
    else:
        # Keep all completion, trim prompt from the left
        prompt_budget = max_length - comp_len
        trim_start = comp_start - prompt_budget

    return {
        key: val[trim_start : trim_start + max_length]
        for key, val in input_dict.items()
    }


def apply_prompt_loss_weight(samples: list[dict], plw: float = 0.05, seed: int = 42) -> list[dict]:
    """Randomly unmask a fraction of prompt tokens for loss computation.

    For datasets with extreme prompt:completion ratios (>10:1), full prompt
    masking (labels=-100) discards all signal about the input language/domain.
    A small PLW gives the model some gradient from prompt tokens — especially
    valuable for non-English data where the tokenizer has low vocab coverage.

    Research: "Instruction Fine-Tuning: Does Prompt Loss Matter?" (2024)
    found PLW=0.01-0.1 significantly helps for short-completion data.
    """
    import random as _rng
    _rng.seed(seed)
    n_unmasked = 0
    for sample in samples:
        input_ids = sample.get("input_ids", [])
        labels = sample.get("labels", [])
        new_labels = list(labels)
        for i, (tok, lab) in enumerate(zip(input_ids, labels)):
            if lab == -100 and _rng.random() < plw:
                new_labels[i] = tok
                n_unmasked += 1
        sample["labels"] = new_labels
    n_total_prompt = sum(1 for s in samples for l in s.get("labels", []) if l == -100)
    print(
        f"[sn56][plw] Unmasked {n_unmasked} prompt tokens (plw={plw}, "
        f"~{n_unmasked / max(1, n_unmasked + n_total_prompt):.1%} of prompt)",
        flush=True,
    )
    return samples


def gaussian_subsample(samples: list[dict], target_size: int, seed: int = 42) -> list[dict]:
    """Subsample dataset with Gaussian weighting centered on median difficulty.

    Difficulty = number of completion tokens (non -100 labels). Oversamples
    medium-difficulty examples, undersamples extremes (trivial / noise).
    Used when dataset is too large to train on fully in the time budget.
    """
    import numpy as _np

    n = len(samples)
    if target_size >= n:
        return samples

    # Score difficulty by completion token count
    diffs = _np.array([
        sum(1 for l in s.get("labels", []) if l != -100) for s in samples
    ], dtype=_np.float64)

    median = _np.median(diffs)
    std = max(_np.std(diffs), 1.0)

    # Gaussian weights: peak at median, tails downweighted
    weights = _np.exp(-0.5 * ((diffs - median) / std) ** 2)
    weights /= weights.sum()

    rng = _np.random.RandomState(seed)
    indices = rng.choice(n, size=target_size, replace=False, p=weights)
    indices.sort()

    print(
        f"[sn56][gauss] {n} -> {target_size} samples "
        f"(median_comp_tokens={median:.0f}, std={std:.0f})",
        flush=True,
    )
    return [samples[i] for i in indices]


def pad_inputs(tokenizer: AutoTokenizer, input_dict: dict, max_length: int, padding_side: str) -> dict:
    assert padding_side in ["left", "right"]
    if max_length <= 0:
        return input_dict
    # Smart truncate: preserve completion tokens, trim prompt side
    input_dict = smart_truncate(input_dict, max_length)
    result = {
        "input_ids": pad_sequence(input_dict["input_ids"], tokenizer.pad_token_id, max_length, padding_side),
        "attention_mask": pad_sequence(input_dict["attention_mask"], 0, max_length, padding_side),
        "labels": pad_sequence(input_dict["labels"], -100, max_length, padding_side),
    }
    return result


class MyDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path: str, max_length: int) -> None:
        super().__init__()
        with open(data_path, 'r') as file:
            self.eval_dataset = json.load(file)

        self.tokenizer = tokenizer
        self.max_length = max_length
        print("padding_side: ", self.tokenizer.padding_side)

    def __len__(self):
        return len(self.eval_dataset)

    def __getitem__(self, idx):
        dp = self.eval_dataset[idx]
        input_dict = pad_inputs(self.tokenizer, dp, self.max_length, self.tokenizer.padding_side)
        for key in input_dict:
            input_dict[key] = torch.tensor(input_dict[key])
        return input_dict