"""Final dev-data pass — data maximization for small datasets.

The dev split is held out only for checkpoint selection; the model that's scored
is graded on the validator's *separate* test set. So at the very end we reclaim
the dev set as a last, gentle training nudge on the best checkpoint, then save.

Safety is structural, not a holdout: we start from the already-selected best
checkpoint and take one low-LR (min_lr) epoch — a bounded perturbation.

DDP — done "properly", reusing the trainer's own machinery rather than a bespoke
loop:
- the EXISTING wrapped model (trainer.model_wrapped) — never re-wrapped, so no
  second set of reducer hooks;
- the dev loader is sharded across ranks via accelerator.prepare (even_batches ->
  every rank sees the same number of micro-batches -> identical step/all-reduce
  pattern -> lockstep, no deadlock);
- gradient accumulation kept at training's value, so the effective batch
  (per_device * grad_accum * world_size) MATCHES training and min_lr is
  calibrated for it (a small unsharded batch would make min_lr effectively too
  hot and risk degrading the model with no eval to catch it);
- the exact accumulate/sync pattern transformers' Trainer uses
  (_set_sync_gradients + no_sync on non-boundary micro-batches,
  accelerator.backward / clip_grad_norm_ on the boundary).

Only rank 0 writes the submission. Best-effort: the caller wraps it so any
failure just leaves the (already-saved) best checkpoint untouched.
"""

import contextlib
import gc
import os
import shutil
from inspect import signature as _signature

import torch
from transformers.trainer_utils import is_main_process


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def run_dev_pass(
    trainer,
    *,
    submission_dir,
    min_lr,
    max_grad_norm,
    train_per_device,
    train_grad_accum,
    local_rank,
    log,
):
    """One low-LR epoch over the trainer's eval (dev) set, then a weights-only
    save into submission_dir. Uses trainer.get_eval_dataloader() so batches are
    built correctly for THIS trainer (plain LM for instruct; tokenized
    chosen/rejected pairs for DPO) and trainer.training_step() so the loss is the
    trainer's own (incl. DPO's reference-model term)."""
    accelerator = trainer.accelerator
    # Reuse the model the trainer already wrapped — do NOT re-wrap (that would
    # register a second set of DDP reducer hooks on the same params).
    ddp_model = getattr(trainer, "model_wrapped", None) or trainer.model
    unwrapped = _unwrap(ddp_model)
    device = next(unwrapped.parameters()).device

    trainable = [p for p in unwrapped.parameters() if p.requires_grad]
    if not trainable:
        log("[sn56][devfit] sem parâmetros treináveis, pulando")
        return

    # Drop the training optimizer + stale grads to free room for a fresh,
    # momentum-free dev optimizer.
    try:
        trainer.optimizer = None
    except Exception:
        pass
    unwrapped.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dev_opt = torch.optim.AdamW(trainable, lr=min_lr, weight_decay=0.0)

    # The trainer's own eval dataloader: correctly processed + sharded for THIS
    # trainer. It uses the eval batch size, so match training's effective batch by
    # accumulating proportionally more micro-steps.
    loader = trainer.get_eval_dataloader()
    per_device_eval = max(1, int(getattr(trainer.args, "per_device_eval_batch_size", 1) or 1))
    accum = max(1, round(train_per_device * train_grad_accum / per_device_eval))

    steps_in_epoch = len(loader)
    # Defensive lockstep: don't trust even_batches — force every rank to run the
    # SAME number of micro-batches (the global min). Identical step counts ->
    # identical boundary/all-reduce pattern -> DDP can't deadlock.
    if torch.distributed.is_initialized():
        _t = torch.tensor([steps_in_epoch], device=device)
        torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.MIN)
        steps_in_epoch = int(_t.item())
    if steps_in_epoch == 0:
        log("[sn56][devfit] dev loader vazio, pulando")
        return

    # training_step normalizes the loss by current_gradient_accumulation_steps.
    try:
        trainer.current_gradient_accumulation_steps = accum
    except Exception:
        pass
    # Some trainers' training_step doesn't take num_items_in_batch (older signature).
    _ts_kwargs = ({"num_items_in_batch": None}
                  if "num_items_in_batch" in _signature(trainer.training_step).parameters
                  else {})

    ddp_model.train()
    n_opt = 0
    for step, inputs in enumerate(loader):
        if step >= steps_in_epoch:   # honor the synced step count on every rank
            break
        do_sync = ((step + 1) % accum == 0) or ((step + 1) == steps_in_epoch)
        # We bypass the Trainer's prefetch-aware loop, so set the flag ourselves.
        accelerator.gradient_state._set_sync_gradients(do_sync)
        sync_ctx = (contextlib.nullcontext() if do_sync
                    else accelerator.no_sync(model=ddp_model))
        with sync_ctx:
            # training_step computes the RIGHT loss for this trainer and handles
            # autocast + _prepare_inputs + accelerator.backward.
            trainer.training_step(ddp_model, inputs, **_ts_kwargs)
        if do_sync:
            if max_grad_norm and max_grad_norm > 0:
                accelerator.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
            dev_opt.step()
            ddp_model.zero_grad(set_to_none=True)
            n_opt += 1

    eff = per_device_eval * accum * max(1, getattr(accelerator, "num_processes", 1))
    log(f"[sn56][devfit] {steps_in_epoch} micro / {n_opt} opt-steps sobre dev "
        f"@ lr={min_lr:.2e} (lote efetivo ~{eff}, igual ao treino)")

    # Free the dev optimizer before the save / any NCCL.
    del dev_opt
    unwrapped.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_main_process(local_rank):
        _save_weights_only(unwrapped, submission_dir, log)

    # Keep the process group aligned for a clean shutdown.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _is_weight_file(name):
    return (name.endswith(".safetensors")
            or name == "model.safetensors.index.json"
            or name.endswith(".bin"))


def _save_weights_only(unwrapped, submission_dir, log):
    """Atomically replace submission_dir with the dev-passed model, keeping its
    already-validated config.json / tokenizer / loss.txt.

    The existing (best) submission is NEVER destroyed until the new one is fully
    built — we stage the whole thing, then a single rename-swap. So if the dev
    pass overruns the deadline and gets killed, the timer's best submission stays
    intact; the dev-passed model only lands if we make it all the way through.
    """
    if not submission_dir or not os.path.isdir(submission_dir):
        log(f"[sn56][devfit] submission_dir ausente ({submission_dir}), não salvo")
        return
    base = submission_dir.rstrip("/")
    staging, backup = base + ".devfit_new", base + ".devfit_old"
    for d in (staging, backup):
        if os.path.exists(d):
            shutil.rmtree(d)
    try:
        os.makedirs(staging, exist_ok=True)
        # 1) dev-passed weights (+ runtime config) into staging
        unwrapped.save_pretrained(staging, safe_serialization=True)
        # 2) overwrite staging's NON-weight files with the validated ones from the
        #    current submission (config/tokenizer/loss.txt/generation_config/...)
        for f in os.listdir(submission_dir):
            src = os.path.join(submission_dir, f)
            if os.path.isfile(src) and not _is_weight_file(f):
                shutil.copy2(src, os.path.join(staging, f))
        # 3) atomic-ish swap: move best aside, move new in, drop best.
        os.rename(submission_dir, backup)
        os.rename(staging, submission_dir)
        shutil.rmtree(backup, ignore_errors=True)
        log("[sn56][devfit] submissão trocada pelo modelo do dev-pass "
            "(config/tokenizer/loss.txt preservados)")
    except Exception as e:
        log(f"[sn56][devfit] falha ao salvar ({e}), restaurando melhor")
        # If the best got moved to backup but the swap didn't finish, restore it.
        if not os.path.isdir(submission_dir) and os.path.isdir(backup):
            try:
                os.rename(backup, submission_dir)
            except Exception:
                pass
        for d in (staging, backup):
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
