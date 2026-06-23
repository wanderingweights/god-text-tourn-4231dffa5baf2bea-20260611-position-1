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
    """Make fla + raven importable before from_pretrained AND ensure raven/ sits
    next to the cached modeling file. QuasarLong.__init__ hard-checks
    os.path.isdir(_HERE/'raven'); under deepspeed zero.Init the model is built
    inside from_pretrained, so the cache must already contain raven/ first."""
    import glob as _glob
    import os as _os
    import shutil as _shutil

    model_dir = Path(model_path).resolve()
    _add_path(model_dir)  # makes `import fla` and `import raven` resolve
    src_raven = model_dir / "raven"
    # Force the remote modeling code into the dynamic-module cache now (resolve the
    # class, do NOT build the model) so the cache dir exists before from_pretrained.
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        get_class_from_dynamic_module(
            "modeling_quasar_long.QuasarLongForCausalLM", str(model_dir), trust_remote_code=True
        )
    except Exception as _e:
        print(f"[quasar] get_class warn: {_e}", flush=True)
    # COPY raven/ next to every cached modeling file. Copy (not symlink) is robust
    # under multi-rank deepspeed; QuasarLong.__init__ checks os.path.isdir(_HERE/raven).
    cache_root = _os.path.join(_os.path.expanduser("~"), ".cache/huggingface/modules/transformers_modules")
    dirs = _glob.glob(_os.path.join(cache_root, "*[Qq]uasar*", "*"))
    print(f"[quasar] prepare: raven_src_exists={src_raven.is_dir()} cache_dirs={dirs}", flush=True)
    for d in dirs:
        if _os.path.isfile(_os.path.join(d, "modeling_quasar_long.py")) and not _os.path.isdir(_os.path.join(d, "raven")):
            _shutil.copytree(src_raven, _os.path.join(d, "raven"), dirs_exist_ok=True)
            print(f"[quasar] copied raven -> {d}/raven", flush=True)


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
    meta_names = [n for n, p in model.named_parameters() if getattr(p, "is_meta", False)]
    if not meta_names:
        return 0
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        index = json.loads(idx_path.read_text())["weight_map"]
    else:
        # Single-file checkpoint (no shard index) — e.g. a warm-start continue-from
        # checkpoint saved as one model.safetensors. Every param lives in that file.
        index = {n: "model.safetensors" for n in meta_names}
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


# Params that are structurally unused in QuasarLong's forward for the
# Quasar-Preview config and therefore NEVER receive a gradient:
#   * branch_local_window_mix_logit — the local-window branch is disabled
#     (hybrid_local_window_size=0), so this mix logit is dead.
#   * quasar_attention.{A_log, dt_bias, f_proj.weight} — quasar-branch SSM params
#     the fla scan path does not backprop into.
# Under full fine-tune + ZeRO-3, deepspeed tries to reduce these params' grads and
# hits `param.grad=None` -> "'NoneType' object has no attribute 'numel'". They do
# not affect the forward, so they cannot be (and need not be) trained: freezing
# them is the correct fix and lets full fine-tune run. Confirmed via grad probe
# (one fwd/bwd) — exactly 46 params, all matching these suffixes.
QUASAR_UNUSED_PARAM_SUFFIXES = (
    "branch_local_window_mix_logit",
    "quasar_attention.A_log",
    "quasar_attention.dt_bias",
    "quasar_attention.f_proj.weight",
)


def freeze_unused_params(model, suffixes=QUASAR_UNUSED_PARAM_SUFFIXES) -> int:
    """Set requires_grad=False on params that get no gradient (see note above).

    Must be called BEFORE deepspeed.initialize so the frozen params are excluded
    from the ZeRO param groups / grad reduction. Returns the count frozen."""
    frozen = 0
    for name, p in model.named_parameters():
        if p.requires_grad and any(name.endswith(s) for s in suffixes):
            p.requires_grad = False
            frozen += 1
    print(f"[quasar] froze {frozen} structurally-unused (no-grad) params for ZeRO-3", flush=True)
    return frozen


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
