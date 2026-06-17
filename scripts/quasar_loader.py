"""Loader helpers for silx-ai/Quasar-Preview (QuasarLongForCausalLM).

This checkpoint is a custom hybrid-attention MoE that needs special handling the
stock loader does not provide:

  * It ships a vendored ``fla/`` package and requires a repo-local ``raven/``
    package (NOT published on the HF repo — fetched separately by the
    downloader). Both must be importable *before* ``from_pretrained`` because
    transformers' trust_remote_code ``check_imports`` resolves ``import fla`` /
    ``import raven`` against ``sys.path``, and the cached dynamic module looks
    for ``raven/`` next to the cached ``modeling_quasar_long.py``.
  * The hybrid quasar/raven/gla branches are ONLY built under
    ``attn_implementation="sdpa"`` (eager / flash_attention_2 silently fall back
    to a dense model and drop the hybrid weights).
  * Some hybrid-branch params come back on the ``meta`` device after load and
    must be filled manually from the safetensors shards.
  * ``tokenizer_config.json`` references a TokenizersBackend that AutoTokenizer
    cannot parse; ``tokenizer.json`` itself is valid.

Mirrors the proven eyad-silx/raven ``generate_quasar.py`` recipe.
"""
import gc
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast


def is_quasar(model_path: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        archs = cfg.architectures or []
        return any("quasarlong" in a.lower() for a in archs)
    except Exception:
        return False


def _add_path(path) -> None:
    path = str(Path(path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def link_raven(model_dir: Path) -> None:
    """Symlink the repo-local raven/ into the model dir (no-op if already there)
    and into every cached transformers dynamic-module dir for this model."""
    raven_pkg = model_dir / "raven"
    if not raven_pkg.is_dir():
        raise FileNotFoundError(
            f"missing {raven_pkg} — the downloader must place the raven package "
            f"inside the model dir (it is not on the Quasar-Preview HF repo)"
        )
    cache_glob = Path.home().glob(
        ".cache/huggingface/modules/transformers_modules/*Quasar*"
    )
    for target_dir in cache_glob:
        if not (target_dir / "modeling_quasar_long.py").exists():
            continue
        link = target_dir / "raven"
        if link.is_symlink() or not link.exists():
            link.unlink(missing_ok=True)
            link.symlink_to(raven_pkg, target_is_directory=True)


def prepare(model_path: str) -> None:
    """Make fla + raven importable before from_pretrained."""
    model_dir = Path(model_path).resolve()
    _add_path(model_dir)  # makes `import fla` and `import raven` resolve
    # Pre-create the dynamic-module cache so the modeling file's _HERE/raven
    # check passes; harmless if the cache is created later (link_raven is also
    # re-run by load_quasar_model after from_pretrained populates the cache).
    try:
        link_raven(model_dir)
    except FileNotFoundError:
        raise
    except Exception:
        pass


def load_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    model_dir = Path(model_path).resolve()
    return PreTrainedTokenizerFast(
        tokenizer_file=str(model_dir / "tokenizer.json"),
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        cls_token="[CLS]",
    )


def _set_param(root, name: str, tensor: torch.Tensor) -> None:
    obj = root
    parts = name.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    old = getattr(obj, parts[-1])
    setattr(
        obj,
        parts[-1],
        torch.nn.Parameter(tensor, requires_grad=getattr(old, "requires_grad", False)),
    )


def fill_meta_params(model, model_path: str) -> int:
    """Fill any params still on the meta device from the safetensors shards.

    The hybrid (sdpa) branches can leave params on meta after from_pretrained.
    Returns the number of params filled."""
    model_dir = Path(model_path).resolve()
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    meta_names = [n for n, p in model.named_parameters() if getattr(p, "is_meta", False)]
    if not meta_names:
        return 0
    by_file = {}
    for name in meta_names:
        by_file.setdefault(index[name], []).append(name)
    for filename, keys in sorted(by_file.items()):
        tensors = load_file(str(model_dir / filename), device="cpu")
        for key in keys:
            _set_param(model, key, tensors[key].to(torch.bfloat16))
        del tensors
        gc.collect()
    return len(meta_names)


def load_quasar_model(model_path: str):
    """Load QuasarLongForCausalLM with hybrid branches enabled (sdpa) and any
    meta params filled. Returns a CPU model; caller moves it to device."""
    prepare(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # REQUIRED: builds quasar/raven/gla branches
        low_cpu_mem_usage=False,
    )
    # from_pretrained populated the dynamic-module cache — (re)link raven there.
    link_raven(Path(model_path).resolve())
    filled = fill_meta_params(model, model_path)
    if filled:
        print(f"[quasar] filled {filled} meta params from safetensors", flush=True)
    return model
