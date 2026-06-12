"""
In-process adaptive learning rate search.

Runs inside the training script after model and data are loaded. Tests
learning rates via short training trials (restore weights between each),
narrowing toward the optimum in three time-gated passes:

  S1  coarse  : 4 LRs around the estimate; ascends the high side only while it
                doesn't diverge, else pivots to lower LRs. S1 gets first claim on
                the budget and goes as deep as it can (the estimate is strong).
  S2  refine  : the S1 winner + 3 neighbours at full depth (skipped if it won't
                fit a real floor — a shallow refine is noisier than the S1 win).
  S3  polish  : opportunistic — extend past an edge winner AND fill the
                widest interior gap, while clock remains.

Budget is epoch-keyed and capped at _SEARCH_FRACTION (20%) of the total:
  <1 epoch  → validate (3-LR sweep) if it fits, else skip to the estimate.
  1 <= e < 2 → two_stage (S1 + S2).   e >= 2 → full (S1 + S2 + S3).
S2/S3 run only if S1 cleared the refine gate AND time remains.

Probe data is cached once and replayed identically across trials (fair
comparison); windows are per-tier and consecutive (refine continues from S1).

Two pruning mechanisms abort hopeless trials early:
  - divergence : rolling loss non-finite or > 1.75x the trial's own best-seen.
  - relative   : a trial is much worse than the best trial seen at the
                 *same* step index (apples-to-apples).

Key properties:
- Single model load — no subprocess overhead per trial.
- Probe weights are DISCARDED — training restarts from the original weights on
  full data, so it never re-sees (over-weights) any probe's batches.
- Loss comparisons only happen within a tier (same step count), so the
  edge-of-stability selection threshold is meaningful.

References:
- Smith 2015: "Cyclical Learning Rates" (LR range test)
- Izmailov et al. 2018: "Averaging Weights Leads to Wider Optima"
- "Stepping on the Edge" (NeurIPS 2024): optimal LR sits at the stability edge
"""

import gc
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch


def _is_oom_error(exception: Exception) -> bool:
    """Check if exception is an out-of-memory or related CUDA error."""
    _statements = [
        " out of memory.",
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
        "DefaultCPUAllocator: can't allocate memory",
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def _clear_memory():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _materialize_batches(loader, n_batches: int) -> list:
    """Pull up to n_batches from `loader` once and keep them on CPU, so every
    trial can replay the SAME data. A fair LR comparison needs identical batches
    across trials, but a fresh DataLoader iterator reshuffles each time. Returns
    a list; trials index it (with a per-tier offset) and cycle past the end on
    small datasets."""
    batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        if isinstance(batch, dict):
            batches.append({
                k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            })
        else:
            batches.append(batch)
    return batches


# ── Search geometry (log10 LR offsets / step counts) ──
_S1_HALF_RANGE = 0.30      # S1 spans estimate ± 0.30 decades (~4x total)
_S1_N = 4                  # 4 coarse LRs in S1 — MUST match the offsets in _do_s1
_S1_MIN_STEPS = 25         # S1 step floor / ceiling (time-determined within)
_S1_MAX_STEPS = 100        # a "really good S1" can go deep (~old-finder horizon) when budget allows
_S1_REFINE_GATE = 50       # min S1 depth before S2 runs — refining a shallow/noisy
                           # S1 winner just polishes noise
_S2_MIN_STEPS = 50         # S2/S3 step floor / ceiling (skip S2 if it won't fit —
_S2_MAX_STEPS = 150        # a too-shallow refine is noisier than the deep S1 winner)
_S2_OFFSETS = (-0.075, 0.075, 0.15)  # 3 neighbours around the S1 winner
_MIN_GAP = 0.04            # don't probe gaps/edges finer than this (~1.1x)
_DEDUP_TOL = 0.02          # treat two log-LRs within this as identical
_MAX_CACHED_BATCHES = 400  # cap on probe batches held in CPU RAM (cycled if exceeded)
_PROBE_WARMUP = 5          # ramp LR 0->target over the first few probe steps (eats INTO
                           # them, not extra) so a high LR isn't judged cold; these ramp
                           # steps are excluded from scoring + pruning

# ── Selection ──
_EDGE_TOLERANCE = 0.025    # pick highest LR whose loss is within 2.5% of best

# ── DPO held-out scoring (search-only; SFT/GRPO unaffected) ──
# DPO's training loss has a DEGENERATE minimiser: a too-high LR collapses the
# policy's log-probs to inflate the chosen−rejected margin, driving TRAINING
# loss on the just-fitted batches toward zero while the model degrades as a
# language model (rewards/chosen craters far below the reference). Selecting on
# training loss therefore picks the most destructive LR. The validator does the
# opposite — it ranks by DPO loss on a HELD-OUT eval set (dpo_trainer.evaluate,
# beta=0.1, base model as reference; lower is better). So for DPO we score each
# probe by held-out DPO loss, mirroring the validator exactly, and select by
# pure argmin (no edge-of-stability bias toward higher LR — that bias points
# straight at the failure mode here).
_DPO_HOLDOUT_BATCHES = 24  # held-out eval batches cached once, replayed per trial
_MAX_EDGE_EXTENSIONS = 4   # validate sweep may step this many LRs past the
                           # winning edge, riding held-out loss to the optimum
                           # (the SFT-scale centre is usually too hot for DPO)

# ── Pruning ──
_PRUNE_WINDOW = 5          # rolling-mean window over recent step losses
_PRUNE_MIN_STEP = 3        # never prune before this many optimizer steps
_PRUNE_DIVERGE_FACTOR = 1.75  # prune only if rolling mean > 1.75x the trial's BEST-seen
_PRUNE_REL_MARGIN = 0.50   # "miles off": > 50% worse than best-at-step-5 → prune.
                           # Deliberately loose — the selection metric is tail
                           # MEDIAN, so a high-but-productive LR that oscillates
                           # early must not be pruned for transient volatility.
_PRUNE_REL_STEP = 5        # relative prune is checked only at this step

# ── Budget ──
_TRAIN_UTIL = 0.85         # fraction of wall clock training actually uses
                           # (matches epoch planning in train_instruct.py)
_SEARCH_FRACTION = 0.20    # the search may spend up to 20% of the total budget
_LEGACY_CAP_SECS = 20 * 60 # absolute cap for the legacy (no steps_per_epoch) path
# The validation sweep (sub-1-epoch jobs) just vets the estimate for divergence,
# so it gets its own cheaper step floor — it doesn't need S1's deeper horizon,
# which keeps the safety net easy to trigger even when per-step cost is high.
_VALIDATE_N = 3            # LRs in the validation sweep (centre, -range, +range)
_VALIDATE_MIN_STEPS = 15   # cheaper floor than _S1_MIN_STEPS for the sweep

_MAX_OOM_RETRIES = 8       # halve batch size up to 8 times (256x reduction)

# ── Warmup-derived curvature ──
# This branch replaces lr_finder's STATIC curvature term (sqrt(ref_loss/init_loss),
# which augmentation contaminates) with one measured from the warmup the search
# already runs. The warmup loss trajectory tells us what a static init_loss can't:
# a fast, smooth drop means we're far from a solution with informative gradients
# (push LR UP — the cold-start regime); a flat or rising trajectory means we're
# near a minimum or already too hot (push DOWN). lr_finder emits a neutral static
# curvature on this branch, so the two never stack.
# Warmup length. On main this only needed to be long enough to time t_per_step
# (5 was plenty); here it is ALSO the curvature signal, so we run a few more steps
# — the factor averages the first/last third of the trajectory and the first 1-2
# steps are atypical (Adam moments + LR ramp warming up). 8 gives ~3 points per
# third at trivial cost (the warmup runs on save/restored weights, then discarded)
# and also tightens the t_per_step estimate. Tune here.
_WARMUP_STEPS = 8
_WC_MIN_POINTS = 3         # need at least this many recorded warmup steps for a signal
_WC_NEUTRAL_DROP = 0.04    # relative loss drop over the warmup window that maps to 1.0x
_WC_GAIN = 4.0             # sensitivity: factor = 1 + GAIN*(rel_drop - NEUTRAL)
_WC_MIN, _WC_MAX = 0.6, 1.6  # clamp the multiplier (bounded like the static term)


def _warmup_curvature_factor(step_losses: list) -> tuple[float, dict]:
    """Map a warmup loss trajectory to an LR multiplier.

    rel_drop = (start - end) / |start|, averaged over the first/last third to
    damp single-step noise. Positive => improving (far from solution => larger
    LR); <= 0 => flat/rising (near solution or too hot => smaller LR). Linear,
    clamped, with one neutral point — deliberately simple; the A/B calibrates it.
    """
    pts = [float(l) for l in step_losses if l is not None and math.isfinite(l)]
    if len(pts) < _WC_MIN_POINTS:
        return 1.0, {"reason": "too_few_points", "n": len(pts)}
    k = max(1, len(pts) // 3)
    l_start = sum(pts[:k]) / k
    l_end = sum(pts[-k:]) / k
    rel_drop = (l_start - l_end) / max(abs(l_start), 1e-6)
    factor = 1.0 + _WC_GAIN * (rel_drop - _WC_NEUTRAL_DROP)
    factor = max(_WC_MIN, min(_WC_MAX, factor))
    return factor, {"l_start": l_start, "l_end": l_end, "rel_drop": rel_drop, "n": len(pts)}


class _BatchChanged(Exception):
    """Raised when an OOM forced a batch halving mid-search. The controller
    rescales the LR centre (sqrt rule), drops the now-incomparable scores, and
    resumes from the refine stage — never back to S1's wide sweep."""


class _SearchExhausted(Exception):
    """Raised when OOM retries are exhausted: give up and use the best so far."""


@dataclass
class BudgetPlan:
    """Outcome of budget planning: how much wall-clock the search may spend."""
    mode: str          # "skip" | "validate" | "two_stage" | "full"
    budget_secs: float
    epochs_affordable: float
    reason: str


def plan_budget(
    total_secs: float,
    steps_per_epoch: Optional[int],
    t_per_step: float,
) -> BudgetPlan:
    """Decide search depth + budget from how many epochs the job can afford.

    The search may spend up to _SEARCH_FRACTION of the total budget; the
    depth scales with the size of the job:

      <1 epoch    → validate: a 3-LR sweep IF it fits that budget, else skip.
      1 <= e < 2  → two_stage: S1 coarse + S2 refine.
      e >= 2      → full: S1 + S2 + S3.

    Every non-skip mode carries the best probe's weights forward ("continue from
    best"), so search steps are kept work, not thrown away. If a job is too tight
    for the depth its epoch-count implies, it degrades to the cheaper sweep, then
    to skip. `epochs_affordable` accounts for _TRAIN_UTIL to match how
    train_instruct.py later sizes the real run.

    When steps_per_epoch is unknown (legacy DPO path), grants a full search
    within the same fraction (capped absolutely).
    """
    search_budget = _SEARCH_FRACTION * total_secs
    _pct = f"{_SEARCH_FRACTION:.0%}"   # log label — never hardcode the number

    if not steps_per_epoch or steps_per_epoch <= 0 or t_per_step <= 0:
        return BudgetPlan(
            mode="full",
            budget_secs=min(search_budget, _LEGACY_CAP_SECS),
            epochs_affordable=float("nan"),
            reason=f"legacy: full search within {_pct} (no steps_per_epoch)",
        )

    epoch_secs = steps_per_epoch * t_per_step
    epochs_affordable = total_secs * _TRAIN_UTIL / epoch_secs
    validate_cost = _VALIDATE_N * _VALIDATE_MIN_STEPS * t_per_step  # cheap sweep floor
    s1_cost = _S1_N * _S1_MIN_STEPS * t_per_step                    # full S1 floor

    def _sweep_or_skip(tag: str) -> BudgetPlan:
        if search_budget >= validate_cost:
            return BudgetPlan(
                mode="validate", budget_secs=search_budget,
                epochs_affordable=epochs_affordable,
                reason=f"{tag} ({epochs_affordable:.2f}), 3-LR sweep within {_pct}",
            )
        return BudgetPlan(
            mode="skip", budget_secs=0.0, epochs_affordable=epochs_affordable,
            reason=f"{tag} ({epochs_affordable:.2f}), sweep won't fit {_pct}",
        )

    if epochs_affordable < 1.0:
        return _sweep_or_skip("<1 epoch")

    if epochs_affordable < 2.0:
        if search_budget >= s1_cost:
            return BudgetPlan(
                mode="two_stage", budget_secs=search_budget,
                epochs_affordable=epochs_affordable,
                reason=f"1<=epochs<2 ({epochs_affordable:.2f}), S1+S2 within {_pct}",
            )
        return _sweep_or_skip("1<=epochs<2 but tight")

    # >= 2 epochs
    if search_budget >= s1_cost:
        return BudgetPlan(
            mode="full", budget_secs=search_budget,
            epochs_affordable=epochs_affordable,
            reason=f">=2 epochs ({epochs_affordable:.2f}), S1+S2+S3 within {_pct}",
        )
    return _sweep_or_skip(">=2 epochs but tight")


def select_edge_lr(
    scores: dict[float, float], tolerance: float = _EDGE_TOLERANCE
) -> tuple[float, float]:
    """Edge-of-stability pick: highest LR whose loss is within `tolerance` of
    the best. Higher LR finds flatter minima and trains faster.

    Args:
        scores: log10(lr) -> loss (lower is better), one tier only.
    Returns:
        (lr, loss) in linear LR space.
    """
    best_loss = min(scores.values())
    threshold = best_loss * (1 + tolerance)
    stable = {lg: ls for lg, ls in scores.items() if ls <= threshold}
    edge_log = max(stable.keys())
    return 10 ** edge_log, stable[edge_log]


def _widest_gap_midpoint(tested_logs: list[float]) -> Optional[float]:
    """Midpoint of the widest interior gap among tested log-LRs, or None if no
    gap exceeds _MIN_GAP."""
    if len(tested_logs) < 2:
        return None
    s = sorted(tested_logs)
    gaps = [(s[i + 1] - s[i], (s[i + 1] + s[i]) / 2) for i in range(len(s) - 1)]
    width, mid = max(gaps, key=lambda g: g[0])
    return mid if width > 2 * _MIN_GAP else None


@torch.no_grad()
def _save_trainable_state(model) -> dict[str, torch.Tensor]:
    """Save all trainable parameters to CPU."""
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data.cpu().clone()
    return state


def _restore_trainable_state(model, state: dict[str, torch.Tensor]) -> None:
    """Restore trainable parameters from saved state."""
    for name, param in model.named_parameters():
        if name in state:
            param.data.copy_(state[name].to(param.device))
            if not param.requires_grad:
                param.requires_grad_(True)


def _run_trial(
    model,
    batches,
    lr: float,
    opt_steps: int,
    optimizer_cls,
    optimizer_kwargs: dict,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    trainer=None,
    use_reward: bool = False,
    prune_fn: Optional[Callable[[int, float, float], Optional[str]]] = None,
    data_offset: int = 0,
    warmup_steps: int = 0,
    eval_fn: Optional[Callable[[], float]] = None,
    out_step_losses: Optional[list] = None,
) -> tuple[float, Optional[str]]:
    """Train for `opt_steps` optimizer steps at `lr`, return (score, prune_reason).

    When eval_fn is given (DPO): score is the held-out DPO loss eval_fn returns
    after training — mirrors how the validator ranks submissions, and is immune
    to the degenerate "minimise training loss by collapsing log-probs" solution.
    When use_reward=False and eval_fn is None (SFT): score is tail-median
    training loss. When use_reward=True (GRPO): score is negative tail-averaged
    reward (negated so lower = better, matching the search's minimization logic).

    If `prune_fn` is given (SFT/DPO only), it is called after every optimizer
    step with (opt_step, rolling_mean_loss, first_step_loss); a non-None return
    aborts the trial early and is propagated as prune_reason.

    If `trainer` is provided, uses trainer.training_step() for the forward pass
    instead of raw model(**batch).
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        # use_reentrant=False is required for LoRA: the default (True) checks
        # if input tensors have requires_grad and skips gradient computation
        # when they don't.  With LoRA the embedding layer is frozen so the
        # first checkpoint block sees no grad-requiring inputs and the entire
        # forward pass produces a loss with no grad_fn.
        gc_kwargs = {"use_reentrant": False}
        if trainer is not None and hasattr(trainer, 'args'):
            gc_kwargs = getattr(trainer.args, 'gradient_checkpointing_kwargs', None) or gc_kwargs
            # Also persist on trainer.args so model.train() (called inside
            # Trainer.training_step) doesn't re-enable with use_reentrant=True.
            trainer.args.gradient_checkpointing_kwargs = gc_kwargs
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_cls(trainable_params, lr=lr, **optimizer_kwargs)

    device = next(model.parameters()).device
    use_amp = device.type == "cuda"

    # Clear trainer metrics so we only capture this trial's rewards
    if use_reward and trainer is not None and hasattr(trainer, '_metrics'):
        trainer._metrics["train"].clear()

    model.train()
    step_losses = []
    micro_loss_accum = 0.0
    micro_count = 0
    total_batches = opt_steps * grad_accum_steps
    prune_reason: Optional[str] = None
    n_cached = len(batches)
    if n_cached == 0:
        return float("inf"), prune_reason
    # Warmup eats into the probe's steps (capped so most stay for scoring).
    warmup = min(warmup_steps, max(0, opt_steps // 3))
    opt_idx = 0

    # Replay the SAME cached batches every trial (deterministic), indexed from
    # this tier's data_offset and cycled if we run past the end (small dataset).
    for i in range(total_batches):
        batch = batches[(data_offset + i) % n_cached]
        if trainer is None:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        try:
            if trainer is not None:
                loss = trainer.training_step(model, batch)
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss, device=device)
            else:
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        outputs = model(**batch)
                        loss = outputs.loss / grad_accum_steps
                else:
                    outputs = model(**batch)
                    loss = outputs.loss / grad_accum_steps
                loss.backward()
        except RuntimeError as e:
            if _is_oom_error(e):
                optimizer.zero_grad()
                _clear_memory()
                raise
            raise

        micro_loss_accum += loss.item() * grad_accum_steps
        micro_count += 1

        if (i + 1) % grad_accum_steps == 0:
            opt_idx += 1
            # Ramp LR 0->target over the first `warmup` steps so a high LR isn't
            # applied cold (matches how training introduces it). No-op when warmup=0.
            for g in optimizer.param_groups:
                g["lr"] = lr * min(1.0, opt_idx / (warmup + 1))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            step_loss = micro_loss_accum / micro_count
            micro_loss_accum = 0.0
            micro_count = 0
            # Warmup steps are INVISIBLE to scoring + pruning: not recorded (so the
            # tail-median and the min-seen prune anchor see only the steady
            # target-LR phase) and not checked (so a high LR gets its ramp + a few
            # real steps before step-3 / step-5 can fire).
            if opt_idx <= warmup:
                continue
            step_losses.append(step_loss)

            # Early pruning (SFT/DPO only — reward dynamics differ).
            if prune_fn is not None and not use_reward:
                opt_step = len(step_losses)
                window = step_losses[-_PRUNE_WINDOW:]
                rolling = sum(window) / len(window)
                prune_reason = prune_fn(opt_step, rolling, step_losses[0])
                if prune_reason is not None:
                    break

    # Expose the recorded trajectory (for warmup-derived curvature) regardless of
    # how the trial is scored below.
    if out_step_losses is not None:
        out_step_losses.extend(step_losses)

    if not step_losses:
        return float("inf"), prune_reason

    # DPO: score on HELD-OUT loss (mirrors the validator's dpo_trainer.evaluate),
    # not training loss on the just-fitted batches. A too-high LR minimises the
    # latter by degenerating; the held-out loss exposes that and ranks LRs the
    # way submissions are actually scored.
    if eval_fn is not None:
        return eval_fn(), prune_reason

    # For GRPO: use reward signal instead of loss
    if use_reward and trainer is not None and hasattr(trainer, '_metrics'):
        rewards = trainer._metrics["train"].get("reward", [])
        if rewards:
            tail_start = max(1, int(len(rewards) * 0.7))
            mean_reward = sum(rewards[tail_start:]) / len(rewards[tail_start:])
            return -mean_reward, prune_reason  # negate: search minimizes

    # SFT/DPO: median of last 20% of step losses.
    # Median is robust to single-step outliers. Last 20% measures where the
    # model settled, not the path it took. Tail AVERAGE penalizes
    # productive-but-volatile high-LR trials; median does not.
    tail_start = max(1, int(len(step_losses) * 0.8))
    tail = sorted(step_losses[tail_start:])
    n = len(tail)
    if n % 2 == 1:
        return tail[n // 2], prune_reason
    return (tail[n // 2 - 1] + tail[n // 2]) / 2, prune_reason


def run_lr_search(
    model,
    train_dataloader,
    initial_lr: float,
    hours_to_complete: float,
    optimizer_cls=None,
    optimizer_kwargs: Optional[dict] = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    dataloader_factory=None,
    trainer=None,
    sync_loss_fn=None,
    use_reward: bool = False,
    steps_per_epoch: Optional[int] = None,
    eval_dataloader=None,
) -> tuple[float, Optional[float]]:
    """Run adaptive LR search in-process (S1 coarse → S2 refine → S3 polish).

    Saves initial weights, runs short trials at different LRs, restores the
    best full-depth trial's weights into the model, and returns the best LR.

    Args:
        model: The model (already on GPU, trainable params set).
        train_dataloader: DataLoader for training data.
        initial_lr: Starting estimate from lr_finder.
        hours_to_complete: Total time budget for the task (hours).
        optimizer_cls: Optimizer class (default: AdamW).
        optimizer_kwargs: Extra kwargs for optimizer (e.g. weight_decay).
        grad_accum_steps: Gradient accumulation steps (match main training).
        max_grad_norm: Gradient clipping norm (match main training).
        dataloader_factory: Rebuilds the loader with a halved batch on OOM.
        trainer: HF Trainer, for DPO/GRPO batch processing.
        sync_loss_fn: Cross-rank loss reducer (keeps ranks' decisions in sync).
        use_reward: GRPO mode — optimise reward, disable pruning.
        steps_per_epoch: Optimizer steps per epoch. When given, the budget is
            anchored to the 1.5-epoch training floor; when None, falls back to
            the legacy fraction-of-total budget.
        eval_dataloader: DPO only — a held-out eval loader. When given (and a
            trainer is present and use_reward is False), each probe is scored by
            DPO loss on a fixed held-out batch set (mirrors the validator) and
            the LR is selected by argmin. SFT (no trainer) and GRPO (use_reward)
            ignore it and keep their training-loss / reward metric.

    Returns:
        (best_lr, t_per_step) — best LR and measured wall-clock per optimizer
        step (from warmup), for use in epoch planning.
    """
    if optimizer_cls is None:
        from torch.optim import AdamW
        optimizer_cls = AdamW
    if optimizer_kwargs is None:
        optimizer_kwargs = {"weight_decay": 0.0}

    # Suppress tqdm progress bar during search if trainer is provided
    _orig_disable_tqdm = None
    if trainer is not None and hasattr(trainer, 'args'):
        _orig_disable_tqdm = getattr(trainer.args, 'disable_tqdm', None)
        trainer.args.disable_tqdm = True

    def _restore_tqdm():
        if trainer is not None and _orig_disable_tqdm is not None:
            trainer.args.disable_tqdm = _orig_disable_tqdm

    device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_secs = hours_to_complete * 3600
    print(
        f"[sn56][farejando] Starting: initial_lr={initial_lr:.2e}, "
        f"device={device}, params={n_params/1e6:.1f}M, "
        f"grad_accum={grad_accum_steps}, max_grad_norm={max_grad_norm}, "
        f"hours={hours_to_complete:.2f}h, steps/epoch={steps_per_epoch}",
        flush=True,
    )

    # ── Warmup: measure t_per_step AND the curvature trajectory (save/restore so
    # it doesn't contaminate the initial state). OOM here triggers batch halving. ──
    print(f"[sn56][farejando] {_WARMUP_STEPS} warmup steps (timing + curvatura)...", flush=True)
    warmup_state = _save_trainable_state(model)
    warmup_oom_retries = 0
    warmup_batches = _materialize_batches(train_dataloader, _WARMUP_STEPS * grad_accum_steps)
    warmup_losses: list = []
    while True:
        try:
            warmup_losses.clear()  # keep only the surviving (non-OOM) attempt
            t0 = time.perf_counter()
            _run_trial(model, warmup_batches, initial_lr, _WARMUP_STEPS, optimizer_cls,
                       optimizer_kwargs, grad_accum_steps, max_grad_norm,
                       trainer=trainer, use_reward=use_reward,
                       out_step_losses=warmup_losses)
            t_warmup = time.perf_counter() - t0
            break
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            _clear_memory()
            warmup_oom_retries += 1
            _restore_trainable_state(model, warmup_state)
            if warmup_oom_retries > _MAX_OOM_RETRIES or dataloader_factory is None:
                print(f"[sn56][farejando] OOM during warmup after {warmup_oom_retries} retries, skipping search", flush=True)
                del warmup_state
                _restore_tqdm()
                return initial_lr, None
            print(f"[sn56][farejando] OOM during warmup, halving batch (retry {warmup_oom_retries}/{_MAX_OOM_RETRIES})", flush=True)
            train_dataloader = dataloader_factory()
            warmup_batches = _materialize_batches(train_dataloader, _WARMUP_STEPS * grad_accum_steps)

    t_per_step = t_warmup / _WARMUP_STEPS
    _restore_trainable_state(model, warmup_state)
    del warmup_state

    # ── Warmup-derived curvature ──
    # Re-centre the search on a curvature factor measured from the warmup we just
    # ran, replacing lr_finder's static term (lr_finder emits neutral curvature on
    # this branch). Bounded by the factor clamp; flows into every downstream path
    # incl. the "skip" early-return below, so it helps tight-budget jobs too.
    wc_factor, wc_diag = _warmup_curvature_factor(warmup_losses)
    if "rel_drop" in wc_diag:
        adjusted = initial_lr * wc_factor
        print(
            f"[sn56][farejando] warmup-curvature: rel_drop={wc_diag['rel_drop']:+.3f} "
            f"(loss {wc_diag['l_start']:.3f}->{wc_diag['l_end']:.3f}, n={wc_diag['n']}) "
            f"-> x{wc_factor:.3f}; lr {initial_lr:.2e} -> {adjusted:.2e}",
            flush=True,
        )
        initial_lr = adjusted
    else:
        print(
            f"[sn56][farejando] warmup-curvature: no signal "
            f"({wc_diag.get('reason')}, n={wc_diag.get('n')}), lr unchanged",
            flush=True,
        )

    # ── Budget planning anchored to the 1.5-epoch floor ──
    plan = plan_budget(total_secs, steps_per_epoch, t_per_step)
    print(
        f"[sn56][farejando] Warmup: {t_per_step:.3f}s/opt_step "
        f"({t_per_step/max(grad_accum_steps,1):.3f}s/batch, accum={grad_accum_steps}); "
        f"budget mode={plan.mode} ({plan.budget_secs/60:.1f}min) — {plan.reason}",
        flush=True,
    )

    if plan.mode == "skip":
        print(f"[sn56][farejando] Skipping search, using estimate lr={initial_lr:.2e}", flush=True)
        _clear_memory()   # free warmup's cache before the caller's NCCL collectives
        _restore_tqdm()
        return initial_lr, t_per_step

    budget_secs = plan.budget_secs
    # Validation is a smaller sweep (centre + two extremes) with its own cheaper
    # floor so it stays easy to trigger; but it's the ONLY stage (no S2/S3
    # follows), so it spends most of its budget here and may probe as DEEP as a
    # refine probe when time allows — it's the real selection on these jobs.
    is_validate = plan.mode == "validate"
    s1_n = _VALIDATE_N if is_validate else _S1_N
    s1_min = _VALIDATE_MIN_STEPS if is_validate else _S1_MIN_STEPS
    s1_max = _S2_MAX_STEPS if is_validate else _S1_MAX_STEPS

    # S1 gets FIRST claim on the budget: size it as deep as the budget allows, up
    # to its ceiling (no fixed fraction). A deep, trustworthy S1 beats three
    # shallow stages — S2/S3 only run on the surplus that's left, and only if S1
    # cleared _S1_REFINE_GATE (see the controller).
    s1_steps = int(budget_secs / (s1_n * t_per_step))
    s1_steps = max(s1_min, min(s1_max, s1_steps))
    # Bail if we cannot even afford S1 at the floor depth.
    if s1_n * s1_min * t_per_step > budget_secs:
        print(
            f"[sn56][farejando] Budget too small for S1 "
            f"(need {s1_n*s1_min*t_per_step:.0f}s, have {budget_secs:.0f}s), "
            f"using estimate lr={initial_lr:.2e}",
            flush=True,
        )
        _clear_memory()   # free warmup's cache before the caller's NCCL collectives
        _restore_tqdm()
        return initial_lr, t_per_step

    # ── Shared trial machinery ──
    initial_state = _save_trainable_state(model)
    log_center = math.log10(initial_lr)
    search_start = time.perf_counter()

    # Per-tier data windows, consecutive: S1 reads [0, s1_batches); the refine
    # tier (S2+S3, compared together) reads the NEXT window — fresh data that
    # continues from S1 rather than repeating it. Cache once so every trial
    # replays identical batches (fair comparison); cycle on small datasets.
    full_data_offset = s1_steps * grad_accum_steps
    _refine_span = 0 if is_validate else _S2_MAX_STEPS  # validate is S1-only
    _cap_batches = min((s1_steps + _refine_span) * grad_accum_steps,
                       _MAX_CACHED_BATCHES)
    probe_batches = _materialize_batches(train_dataloader, _cap_batches)
    print(f"[sn56][farejando] cacheei {len(probe_batches)} lotes p/ provas "
          f"(S1@0, refino@{full_data_offset}); mesma data por etapa", flush=True)

    # ── DPO held-out scoring (search-only) ──
    # When a held-out eval loader is supplied (DPO; trainer present, not GRPO),
    # cache a FIXED held-out set once and score every probe on it via the same
    # trainer.compute_loss the validator's evaluate() uses — beta, reference
    # model and loss are identical, so the search optimises the exact quantity
    # submissions are ranked by. SFT (no trainer) and GRPO (use_reward) skip this
    # and keep their existing metric.
    _eval_scored = (
        eval_dataloader is not None and trainer is not None and not use_reward
    )
    eval_fn: Optional[Callable[[], float]] = None
    if _eval_scored:
        holdout_batches = _materialize_batches(eval_dataloader, _DPO_HOLDOUT_BATCHES)
        if not holdout_batches:
            print("[sn56][farejando] sem lotes de validação — caio no loss de treino", flush=True)
            _eval_scored = False
        else:
            def eval_fn() -> float:
                """Mean DPO loss over the held-out set for the current weights."""
                was_training = model.training
                model.eval()
                total, n = 0.0, 0
                try:
                    with torch.no_grad():
                        for b in holdout_batches:
                            prepared = trainer._prepare_inputs(b)
                            loss = trainer.compute_loss(model, prepared)
                            if isinstance(loss, tuple):
                                loss = loss[0]
                            total += float(loss.item())
                            n += 1
                finally:
                    if was_training:
                        model.train()
                return total / n if n else float("inf")
            print(f"[sn56][farejando] score = DPO loss held-out em "
                  f"{len(holdout_batches)} lotes (igual ao validador); seleção=argmin",
                  flush=True)

    # tier scores: log_lr -> loss (only non-pruned trials recorded here)
    s1_scores: dict[float, float] = {}
    full_scores: dict[float, float] = {}
    tested_logs: set[float] = set()       # every log_lr we have run (dedup)
    best_full_log: Optional[float] = None  # lowest-loss full trial (rescale anchor)
    best_full_loss = float("inf")
    best_s1_loss = float("inf")            # lowest-loss S1 trial (for the *** marker)
    best_at_checkpoint: dict[int, float] = {}  # step -> best rolling mean (rel prune)

    oom_state = {"retries": 0, "loader": train_dataloader, "batches": probe_batches}
    step_timer = {"secs": 0.0, "steps": 0}  # steady-state per-step time from real probes

    def _make_prune_fn() -> Callable[[int, float, float], Optional[str]]:
        trial_min = {"v": float("inf")}   # best rolling mean this trial has reached
        def prune_fn(opt_step: int, rolling: float, first: float) -> Optional[str]:
            if math.isfinite(rolling):
                trial_min["v"] = min(trial_min["v"], rolling)
            if opt_step < _PRUNE_MIN_STEP:
                return None
            # Divergence: anchor to the trial's BEST-seen (min rolling), not the
            # noisy first step — a high, edge-of-stability LR often bumps early
            # then descends, so judge it against its real progress and only kill a
            # genuine blow-up (> _PRUNE_DIVERGE_FACTOR x its best).
            base = trial_min["v"] if math.isfinite(trial_min["v"]) else first
            if not math.isfinite(rolling) or rolling > base * _PRUNE_DIVERGE_FACTOR:
                return f"diverge(roll={rolling:.3f} vs min={base:.3f})"
            # Relative "miles off": only at step 5, only vs best seen at step 5.
            if opt_step == _PRUNE_REL_STEP:
                prev = best_at_checkpoint.get(_PRUNE_REL_STEP)
                if prev is not None and rolling > prev * (1 + _PRUNE_REL_MARGIN):
                    return f"rel(step5 roll={rolling:.3f} vs best={prev:.3f})"
                if prev is None or rolling < prev:
                    best_at_checkpoint[_PRUNE_REL_STEP] = rolling
            return None
        return prune_fn

    def _already_tested(log_lr: float) -> bool:
        return any(abs(log_lr - t) < _DEDUP_TOL for t in tested_logs)

    # Mutable search state the OOM handler rescales on a batch change.
    refine_center = log_center   # centre S2 refines around (S1 winner, or rescaled best post-OOM)
    s2_steps = _S2_MIN_STEPS     # shared by S2 and S3

    def _on_batch_halved() -> None:
        """One OOM halving: rescale the LR centre (sqrt rule), drop the now-
        incomparable old-batch scores, shrink the per-step time estimate. Best
        weights are kept as carry-forward; their loss is reset so new-batch
        trials repopulate the comparison set."""
        nonlocal log_center, refine_center, t_per_step, best_full_loss
        # Anchor for the rescale, in the CURRENT batch scale. Trust best_full_log
        # only while full_scores is non-empty (i.e. it's from this generation);
        # both are cleared on each halving, so on consecutive halvings we fall
        # through to log_center — which already carries the prior shifts — and the
        # rescale compounds correctly instead of re-shifting a stale anchor once.
        if best_full_log is not None and full_scores:
            best_known = best_full_log
        elif s1_scores:
            best_known = min(s1_scores, key=s1_scores.get)
        else:
            best_known = log_center
        shift = math.log10(math.sqrt(0.5))   # sqrt LR-batch rule, one halving
        log_center = best_known + shift
        refine_center = log_center
        s1_scores.clear()
        full_scores.clear()
        best_at_checkpoint.clear()
        tested_logs.clear()
        best_full_loss = float("inf")        # redo losses at the new batch scale
        t_per_step *= 0.5                    # per-step time ~halves with batch (estimate)
        step_timer["secs"] = 0.0             # old-batch step timings no longer represent
        step_timer["steps"] = 0              # the (smaller) current batch — re-measure
        print(
            f"[sn56][farejando] OOM→lote/2: recentro lr={10**log_center:.2e} "
            f"(shift {shift:+.3f}), placar limpo, retomando refino",
            flush=True,
        )

    def test_lr(log_lr: float, opt_steps: int, tier: str) -> Optional[float]:
        """Run one trial; record into the tier's score dict. Returns the loss,
        or None if pruned. Raises _BatchChanged on an OOM halving and
        _SearchExhausted when OOM retries run out."""
        # Dedup is per-depth: S1 (coarse) and full (S2/S3) measure the same LR
        # at different step counts, so a full-tier trial must NOT be skipped just
        # because S1 already ran that LR. Only full-tier points populate
        # tested_logs; this lets S2 re-measure the S1 winner at full depth.
        if tier != "s1" and _already_tested(log_lr):
            return None
        lr = 10 ** log_lr
        # Pruning makes per-rank step counts diverge within a trial — only safe
        # when the trial body has NO cross-rank collectives (raw instruct path).
        # With a trainer (DPO) or sync_loss_fn, keep fixed-length trials to avoid
        # an NCCL deadlock. GRPO (use_reward) is trainer-based, also excluded.
        _prune_safe = not (use_reward or trainer is not None or sync_loss_fn is not None)
        prune_fn = _make_prune_fn() if _prune_safe else None

        _restore_trainable_state(model, initial_state)
        # Per-tier data window: S1 reads from 0; the refine tier (full) reads the
        # NEXT window, continuing from where S1's data ended.
        _offset = 0 if tier == "s1" else full_data_offset
        _call_t0 = time.perf_counter()
        try:
            loss, pruned = _run_trial(
                model, oom_state["batches"], lr, opt_steps,
                optimizer_cls, optimizer_kwargs, grad_accum_steps,
                max_grad_norm, trainer=trainer, use_reward=use_reward,
                prune_fn=prune_fn, data_offset=_offset,
                warmup_steps=_PROBE_WARMUP, eval_fn=eval_fn,
            )
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            _clear_memory()
            oom_state["retries"] += 1
            if oom_state["retries"] > _MAX_OOM_RETRIES or dataloader_factory is None:
                print(f"[sn56][farejando] OOM após {oom_state['retries']} tentativas, encerrando busca", flush=True)
                raise _SearchExhausted from e
            print(f"[sn56][farejando] OOM em lr={lr:.2e}, lote/2 "
                  f"(tentativa {oom_state['retries']}/{_MAX_OOM_RETRIES})", flush=True)
            _restore_trainable_state(model, initial_state)
            oom_state["loader"] = dataloader_factory()
            oom_state["batches"] = _materialize_batches(
                oom_state["loader"], len(oom_state["batches"]))
            _on_batch_halved()
            raise _BatchChanged from e

        _call_dt = time.perf_counter() - _call_t0
        if sync_loss_fn is not None:
            loss = sync_loss_fn(loss)

        if tier != "s1":           # only full-tier points gate dedup (see above)
            tested_logs.add(log_lr)
        elapsed = time.perf_counter() - search_start
        metric = f"reward={-loss:.4e}" if use_reward else f"loss={loss:.4e}"

        if pruned is not None:
            print(f"[sn56][farejando] {tier.upper()} [{elapsed:5.0f}s] "
                  f"lr={lr:.2e} {metric} PRUNED {pruned}", flush=True)
            return None

        # Real steady-state step time from a COMPLETED probe (post-warmup, no
        # cold first step), amortized across probes — returned for epoch planning
        # so the cosine covers training even after pruning eats variable time.
        step_timer["secs"] += _call_dt
        step_timer["steps"] += opt_steps

        nonlocal best_full_log, best_full_loss, best_s1_loss
        (s1_scores if tier == "s1" else full_scores)[log_lr] = loss
        marker = ""
        # Track the best loss per tier (for the *** marker, and for full the
        # rescale anchor). We do NOT snapshot the weights: carrying them into the
        # run would make it re-see — and over-weight — the probe's data. The
        # search's product is the LR; training starts clean on full data.
        if tier == "s1":
            if loss < best_s1_loss:
                best_s1_loss = loss
                marker = " *** BEST"
        elif loss < best_full_loss:
            best_full_loss = loss
            best_full_log = log_lr
            marker = " *** BEST"
        print(f"[sn56][farejando] {tier.upper()} [{elapsed:5.0f}s] "
              f"lr={lr:.2e} {metric} ({opt_steps}st){marker}", flush=True)
        return loss

    def _time_left() -> float:
        return budget_secs - (time.perf_counter() - search_start)

    # ── Stage bodies ──
    def _extend_validate_edges() -> None:
        """DPO held-out sweep only: the SFT-scale centre is usually too hot for
        full-weight DPO, so the argmin tends to sit at the LOW edge. Step further
        past whichever edge currently wins the held-out loss until an interior
        minimum appears (or the extension / time budget runs out) — riding the
        held-out curve down to the optimum instead of stopping at a bad bracket."""
        for _ in range(_MAX_EDGE_EXTENSIONS):
            if not s1_scores:
                return
            # Rough cost guard: one more probe (s1_steps) + its held-out eval.
            if _time_left() <= (s1_steps + _DPO_HOLDOUT_BATCHES) * t_per_step:
                return
            tested = sorted(s1_scores)
            best_log = min(s1_scores, key=s1_scores.get)
            if best_log <= tested[0] + 1e-9:
                nxt = tested[0] - _S1_HALF_RANGE       # argmin at low edge → go lower
            elif best_log >= tested[-1] - 1e-9:
                nxt = tested[-1] + _S1_HALF_RANGE      # argmin at high edge → go higher
            else:
                return                                  # interior minimum → done
            if any(abs(nxt - k) < _DEDUP_TOL for k in s1_scores):
                return
            test_lr(nxt, s1_steps, "s1")

    def _do_s1() -> None:
        if plan.mode == "validate":
            # Smaller sweep: just vet the estimate — centre + the two extremes.
            print(f"[sn56][farejando] validate-S1: 3 LRs @ {s1_steps}st, "
                  f"±{_S1_HALF_RANGE} dec (centro lr={10**log_center:.2e})", flush=True)
            for off in (0.0, -_S1_HALF_RANGE, _S1_HALF_RANGE):
                test_lr(log_center + off, s1_steps, "s1")
            # DPO (held-out scored): ride the held-out loss past the winning edge.
            if _eval_scored:
                _extend_validate_edges()
            return
        # Full S1 (_S1_N probes): always probe the estimate + the low edge, then
        # ASCEND the high side only while it doesn't diverge. Once a higher LR
        # blows up, climbing further is pointless — pivot the remaining slot(s) to
        # LOWER LRs instead (the estimate is frequently too hot and the winner
        # often sits at the low edge).
        print(f"[sn56][farejando] S1: {_S1_N} LRs @ {s1_steps}st, "
              f"±{_S1_HALF_RANGE} dec (centro lr={10**log_center:.2e})", flush=True)
        ran = 0
        for off in (0.0, -_S1_HALF_RANGE):
            test_lr(log_center + off, s1_steps, "s1"); ran += 1
        highs = (_S1_HALF_RANGE / 2, _S1_HALF_RANGE)          # +0.15, +0.30
        lows = (-_S1_HALF_RANGE / 2, -_S1_HALF_RANGE * 1.5)   # -0.15 (interior gap) then -0.45
        diverged = False
        for off in highs:
            if ran >= _S1_N:
                break
            r = test_lr(log_center + off, s1_steps, "s1"); ran += 1
            if r is None:            # high LR diverged → stop climbing
                diverged = True
                break
        if diverged:
            print("[sn56][farejando] LR alta divergiu — pivô pra baixo", flush=True)
            for off in lows:
                if ran >= _S1_N:
                    break
                test_lr(log_center + off, s1_steps, "s1"); ran += 1

    def _do_s2() -> None:
        nonlocal s2_steps
        s2_steps = int(_time_left() * 0.6 / (4 * t_per_step))
        s2_steps = max(_S2_MIN_STEPS, min(_S2_MAX_STEPS, s2_steps))
        # Need a real depth floor for the refine to be meaningful — skip S2 (and
        # leave the time) rather than run a too-shallow, noisy refine.
        if _time_left() <= 4 * _S2_MIN_STEPS * t_per_step:
            print(f"[sn56][farejando] Sem tempo pra S2 ({_time_left():.0f}s), só S1", flush=True)
            return
        print(f"[sn56][farejando] S2: melhor + {len(_S2_OFFSETS)} vizinhos @ {s2_steps}st "
              f"(centro lr={10**refine_center:.2e})", flush=True)
        test_lr(refine_center, s2_steps, "full")   # re-measure centre at depth
        for off in _S2_OFFSETS:
            test_lr(refine_center + off, s2_steps, "full")

    def _do_s3() -> None:
        s3_steps = s2_steps
        while full_scores and _time_left() > _S2_MIN_STEPS * t_per_step:
            best_log = min(full_scores, key=full_scores.get)
            lo, hi = min(full_scores), max(full_scores)
            candidates: list[float] = []
            if best_log <= lo + 1e-9:            # extend past a low edge
                candidates.append(best_log - _S1_HALF_RANGE / 2)
            if best_log >= hi - 1e-9:            # extend past a high edge
                candidates.append(best_log + _S1_HALF_RANGE / 2)
            mid = _widest_gap_midpoint(list(full_scores.keys()))
            if mid is not None:                  # fill the widest interior gap
                candidates.append(mid)
            candidates = [c for c in candidates if not _already_tested(c)]
            if not candidates:
                break
            print(f"[sn56][farejando] S3: polir {len(candidates)} pt(s) @ {s3_steps}st", flush=True)
            progressed = False
            for c in candidates:
                if _time_left() <= _S2_MIN_STEPS * t_per_step:
                    break
                if test_lr(c, s3_steps, "full") is not None:
                    progressed = True
            if not progressed:
                break

    # ── Stage controller. A mid-search OOM (_BatchChanged) rescales the centre
    # and resumes from the refine stage — never back to S1's wide sweep — using
    # whatever wall-clock remains (elapsed is never reset). Bounded by the OOM
    # retry cap, which raises _SearchExhausted to stop with the best so far. ──
    stage = 1
    while True:
        try:
            if stage <= 1:
                _do_s1()
                if not s1_scores:
                    break  # all pruned/failed → selection falls back to estimate
                refine_center = min(s1_scores, key=s1_scores.get)
                stage = 2
                # Re-estimate step time from S1's REAL steps (post-warmup) before
                # the S2 go/no-go: the warmup over-estimates, which would wrongly
                # starve the refine stage. S2/S3 sizing + skip-check then use it.
                if step_timer["steps"] >= 10:
                    t_per_step = step_timer["secs"] / step_timer["steps"]
            if plan.mode in ("two_stage", "full") and s1_steps >= _S1_REFINE_GATE:
                if stage <= 2:
                    _do_s2()
                    stage = 3
                if plan.mode == "full":
                    _do_s3()
            elif plan.mode in ("two_stage", "full"):
                print(f"[sn56][farejando] S1 raso ({s1_steps}st < porta "
                      f"{_S1_REFINE_GATE}st): sem refino, fica o melhor do S1", flush=True)
            break
        except _BatchChanged:
            stage = 1 if stage == 1 else 2   # don't go back to S1 once refining
            continue
        except _SearchExhausted:
            break

    # ── Selection: edge rule within the deepest populated tier. ──
    pool = full_scores if full_scores else s1_scores
    if not pool:
        # Nothing survived (all pruned, or OOM cleared then exhausted): fall back
        # to the current (possibly rescaled) centre estimate.
        best_lr, best_loss = 10 ** log_center, float("inf")
        print(f"[sn56][farejando] Sem resultados, estimativa lr={best_lr:.2e}", flush=True)
    elif use_reward:
        best_log = min(pool, key=pool.get)
        best_lr, best_loss = 10 ** best_log, pool[best_log]
    elif _eval_scored:
        # Validator ranks by lowest held-out DPO loss — match it exactly. No
        # edge-of-stability bias toward higher LR (that is the degeneration
        # direction for DPO and is what blew the model up last time).
        best_log = min(pool, key=pool.get)
        best_lr, best_loss = 10 ** best_log, pool[best_log]
    else:
        best_lr, best_loss = select_edge_lr(pool)

    # Always restore the ORIGINAL weights — never carry probe weights into the
    # run. Continuing from them would make training re-see (and over-weight) the
    # probe's data, and we never want to repeat data. The probe's value is the LR.
    _restore_trainable_state(model, initial_state)
    del initial_state

    # Release the trials' cached GPU blocks (last optimizer + activations).
    # The caller's post-search NCCL collectives (batch-size all_reduce, lr +
    # weight broadcasts) and the training optimizer need headroom — NCCL
    # allocates OUTSIDE torch's caching allocator and OOMs if the cache holds
    # all of VRAM, which is fatal (kills the whole process group).
    try:
        model.zero_grad(set_to_none=True)   # drop any lingering grads first
    except Exception:
        pass
    _clear_memory()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Diagnostic anchor: if this line is ABSENT from a run's logs, the
        # trainer is executing stale code (this OOM-cleanup commit isn't deployed).
        print(
            f"[sn56][farejando] limpeza pós-busca: "
            f"{torch.cuda.mem_get_info()[0]/1e9:.1f}GB livres / "
            f"{torch.cuda.memory_reserved()/1e9:.1f}GB reservados por torch",
            flush=True,
        )

    # Refine t_per_step from the REAL probe steps (steady-state, post-warmup):
    # the warmup estimate has a cold first step and over-estimates step
    # cost, which makes epoch planning under-cover training. Returned to the
    # caller, which recomputes remaining wall-clock (post-search) and sizes the
    # cosine from it — so coverage holds even after pruning ate variable time.
    # Fall back to the warmup value if too few steps were timed (all pruned).
    if step_timer["steps"] >= 10:
        measured_t = step_timer["secs"] / step_timer["steps"]
        print(f"[sn56][farejando] t/passo refinado {measured_t:.3f}s "
              f"(warmup era {t_per_step:.3f}s; {step_timer['steps']} passos cronometrados)",
              flush=True)
        t_per_step = measured_t

    _restore_tqdm()
    elapsed = time.perf_counter() - search_start
    tier_name = "full" if full_scores else ("S1" if s1_scores else "estimate")
    print(
        f"[sn56][farejando] Done in {elapsed:.0f}s ({elapsed/60:.1f}min): "
        f"best_lr={best_lr:.2e} (loss={best_loss:.4e}, tier={tier_name}, "
        f"{len(s1_scores)} S1 + {len(full_scores)} full kept)",
        flush=True,
    )
    return best_lr, t_per_step
