"""Competition quiet-mode gate.

Import this module as the *very first* import in every process entrypoint
(``text_trainer``, the ``train_*`` and ``tokenize_*`` scripts, ...) — before any
heavy library import — e.g.::

    import quiet_mode  # noqa: F401,E402  (must run before transformers/datasets)

Importing first matters because the verbosity knobs below are env vars the
libraries read *at import time*; setting them after ``import transformers`` is
too late. Each pipeline stage is a fresh interpreter (``python -m text_trainer``
spawns ``python train_instruct.py`` / ``tokenize_*.py`` subprocesses), so the
gate must be imported in each entrypoint. The print-shim and logging changes are
process-global, so once imported in the entrypoint they cover every module that
entrypoint pulls in.

Quiet mode is ON BY DEFAULT (so competition logs never leak by accident). It
silences the noisy info/strategy chatter that would otherwise leak hints in
publicly-visible logs:
  * ``print()`` to stdout becomes a no-op (the ~300 scattered strategy prints)
  * library log levels are forced to ERROR (transformers/datasets/vllm/...)
  * tqdm progress bars are disabled

Errors are deliberately preserved: explicit ``print(..., file=sys.stderr)`` and
uncaught-exception tracebacks (which go to stderr via sys.excepthook) still
surface, so a crash is still diagnosable from the shared logs and the existing
OOM-retry detection in text_trainer keeps working.

To get full logs back for local debugging, opt out with either
``SN56_VERBOSE=1`` or ``SN56_QUIET=0`` in the environment. This module must
NEVER raise — an exception here would break the import of every entrypoint, so
everything is wrapped defensively.
"""

import builtins
import functools
import logging
import os
import sys


_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")

# Captured once, at module import, before we swap builtins.print below.
_real_print = builtins.print


def _quiet_print(*args, **kwargs):
    # Preserve anything explicitly aimed at stderr (error reporting) and let
    # tracebacks through; drop normal stdout chatter.
    if kwargs.get("file", None) is sys.stderr:
        return _real_print(*args, **kwargs)
    return None


# Make the shim impersonate builtins.print (__name__/__module__/__wrapped__/...).
# Some libraries introspect the print global at import time — notably numba's
# ``@infer_global(print)`` asserts
#     getattr(sys.modules[print.__module__], print.__name__) is print
# A bare nested function reports __module__="quiet_mode", __name__="_quiet_print",
# which is NOT a module attribute -> AttributeError -> numba (hence axolotl, hence
# every text-training run) fails to import. Impersonating print makes that lookup
# resolve to ``getattr(builtins, "print")``, which is this shim. Guarded so the
# impersonation can never raise at import and violate the never-raise invariant.
try:
    functools.update_wrapper(_quiet_print, _real_print)
except Exception:
    pass


def _is_quiet() -> bool:
    # Quiet by default. Opt out (full logs) via SN56_VERBOSE=1 or SN56_QUIET=0,
    # supporting both mental models — an explicit "be verbose" and an explicit
    # "turn quiet off".
    if os.environ.get("SN56_VERBOSE", "").strip().lower() in _TRUTHY:
        return False
    if os.environ.get("SN56_QUIET", "").strip().lower() in _FALSY:
        return False
    return True


def _silence_libraries() -> None:
    # Set verbosity env vars BEFORE the libraries are imported (sitecustomize
    # runs first), so they pick these up at import time. Use setdefault so an
    # explicit override in the environment still wins.
    env_defaults = {
        "TQDM_DISABLE": "1",                 # kills all tqdm progress bars
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "DATASETS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",   # also removes the fork warning
        "VLLM_LOGGING_LEVEL": "ERROR",
        "BITSANDBYTES_NOWELCOME": "1",
        "ACCELERATE_LOG_LEVEL": "error",
        "DATASETS_DISABLE_PROGRESS_BARS": "1",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)

    logging.disable(logging.WARNING)  # drop everything below ERROR globally
    logging.getLogger().setLevel(logging.ERROR)
    for name in (
        "transformers",
        "datasets",
        "accelerate",
        "deepspeed",
        "vllm",
        "axolotl",
        "torch",
        "huggingface_hub",
        "filelock",
        "urllib3",
    ):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass


def _silence_print() -> None:
    builtins.print = _quiet_print


def _main() -> None:
    if not _is_quiet():
        return
    try:
        _silence_libraries()
    except Exception:
        pass
    try:
        _silence_print()
    except Exception:
        pass


_main()
