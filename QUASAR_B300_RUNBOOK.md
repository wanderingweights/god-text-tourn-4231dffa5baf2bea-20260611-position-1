# Quasar-Preview on B300 — What We Did + New-Trainer Runbook

_Last updated: 2026-06-22. Branch: `quasar-ddp` (fork `wanderingweights/god-text-tourn-...`)._

**TL;DR** — silx-ai/Quasar-Preview (18B hybrid-attention MoE) full fine-tunes on a **single
NVIDIA B300** (Blackwell, sm_103) under this trainer: packed, ~**1,600 real tok/s** (~3× the
unpacked baseline), checkpoints auto-push to HF for crash-safety. The remaining gap to silx's
3k-tok/s target is **multi-GPU** (free per-GPU memory → bigger batch → feed the 256-expert MoE),
not more single-card tuning.

---

## Part 1 — What we did

### Goal
Make Quasar-Preview (`QuasarLongForCausalLM`; 18B total / ~2B active; 256-expert MoE; hybrid
quasar/raven/gla layers 4–19; NoPE long-context, `max_position_embeddings=5e6`; needs
`attn_implementation="sdpa"` + vendored `fla/` + repo-local `raven/`) trainable **and fast** under
the GOD/SN56 text-tournament trainer, full fine-tune (no LoRA).

### 1. B300 / Blackwell (sm_103) bring-up — the hard part
Blackwell Ultra has no out-of-the-box support in the torch-2.7 stack. Working recipe (baked into
`dockerfiles/standalone-text-trainer.dockerfile`):
- **torch cu126 → cu128** (cu126 has no Blackwell kernels).
- **triton 3.7.1** (`pip install --pre triton==3.7.1`). torch-2.7.1's bundled triton 3.3.1 can't
  codegen sm_103 (LLVM "cannot select shfl.sync"); 3.5 fixes shfl but can't lower the Blackwell
  5th-gen tensor-core MMA (`tcgen05.*`); **3.7.1 does**. (Hits any Blackwell GPU, not just B300.)
- **CUDA 12.9 ptxas** via `pip install nvidia-cuda-nvcc-cu12` +
  `ENV TRITON_PTXAS_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cuda_nvcc/bin/ptxas`
  (triton's bundled ptxas predates `sm_103a`).
- **Dropped flash-attn** from the image (Quasar uses sdpa; FA is unused and risky to compile on
  Blackwell). Mount `/root/.triton` from the host to persist the kernel cache across restarts
  (skips the ~5-min cold compile).

### 2. Full fine-tune works
- `train_instruct.py` freezes the **46 structurally-unused params** (`branch_local_window_mix_logit`
  + quasar `A_log/dt_bias/f_proj`) so the optimizer doesn't choke on `grad=None`.
- Fixed a DDP-path crash: `len(training_args.fsdp)` → `bool(getattr(training_args,"fsdp",None))`
  (transformers 5 defaults `fsdp=None`).

### 3. Sample-packing — the throughput win
Quasar data is variable-length (mean ~664, full data to ~15k tokens), so padding short samples to a
fixed length wasted ~67% of compute. **Packing** fills a fixed buffer with multiple real sequences.
- **Model side** (silx + a packing agent; repo `wanderingweights/quasar-packing`): added per-segment
  isolation under sdpa — a **block-diagonal attention mask** + **per-segment `cu_seqlens`**, both
  derived from `position_ids` that reset at each segment start. Verified: packed-vs-unpacked loss
  identical, zero cross-segment leakage. (3 edited files: `modeling_quasar_long.py`,
  `raven/layers/raven.py`, `fla/layers/quasar.py`.)
- **Data side** (this repo, `train_instruct.py`): enabled the existing **naive packer**
  (`monkeypatch.pack_data_points_naive`) for Quasar — it already emits the exact contract
  (concat + `position_ids` reset/segment + `labels[0]=-100`/segment + trailing pad). `packing=True`,
  `packing_mode="naive"`, buffer `max_length=4096`; collator preserves `position_ids` + tallies real
  tokens.
- **Contract the data side must emit** (`position_ids` is the *sole* isolation signal — no new
  kwargs): `position_ids` [B,S] reset to 0 each segment start; `attention_mask` 2D 1/0 with TRAILING
  pad; `labels` = `-100` on pad **and** wherever `position_ids==0` (else segment k trains to predict
  segment k+1's first token across the boundary).
- **Result**: 88% real tokens (vs 33%), ~1,600 real tok/s (vs ~530), loss healthy.

### 4. LR-finder memory fix
The empirical LR search (`lr_search.run_lr_search`) defaulted its trial optimizer to **fp32 AdamW**
(144 GB of m+v for 18B) while training uses **`paged_adamw_8bit`** (~36 GB). That 4× silently OOM'd
the search at bs=4 and force-halved the batch (→ half throughput). Fix: pass training's optimizer to
the search (`optimizer_cls=bitsandbytes.optim.PagedAdamW8bit`). It already accepted `optimizer_cls`;
it was just never passed. (Also makes the found LR calibrated on the real optimizer.)

### 5. The save bug (would have blocked every real run)
`save_pretrained → remove_tied_weights_from_state_dict` crashed on **every** checkpoint:
`AttributeError: 'list' object has no attribute 'keys'`. The model declares
`_tied_weights_keys = ["lm_head.weight"]` (old **list** format); transformers 5.x expects a **dict**.
`tie_word_embeddings=false` (untied), so `load_model` now clears the list-form attr on all modules →
save skips tied-weight removal and writes all weights. **Training would otherwise run for hours then
crash at the first save**, with the "add noise" fallback also failing (it doesn't put fla/raven on
`sys.path`). Found via the save-verification run.

### 6. Checkpoint → HF crash-safety
Direct `text_trainer` runs bypass `job_handler`'s HF upload (that's the trainer-service path), so
saves were local-only. Added a `_HFCkptUpload` `TrainerCallback`: on each save it `upload_folder`s the
checkpoint (model files only) to `HF_CKPT_REPO`, latest-only/overwrite. Repo + token from env
(`HF_CKPT_REPO`, `HUGGINGFACE_TOKEN`) — no secret in-script; no-op if unset; never raises.
**Verified**: `checkpoint-3` pushed to `gradients-io-tournaments/quasar_test_run` (185s for ~36 GB).

### 7. Throughput & the path to 3k
~1,600 real tok/s on one B300 is **near the single-card ceiling**: 18B *full*-FT needs ~110–180 GB
just for weights+grads+8-bit-optimizer, leaving little for activations → bs≈4 at the 4096 buffer →
the 256-expert MoE is under-fed (memory-bound, compute-light). Padding is solved (88% real). **3k
needs multi-GPU** (ZeRO/FSDP shard the optimizer/grads → free per-GPU memory → bigger batch → MoE
saturates), or fp8 on the Blackwell tensor cores (kernel work, the B300's headline feature we're not
using yet).

---

## Part 2 — Runbook: stand up a new trainer + run Quasar

### 0. Box
A single big-memory Blackwell card (B300 ~275 GB; B200 ~192 GB also works). 18B full-FT needs ~250 GB
resident, so an 80 GB H100 will NOT fit single-card (needs ZeRO sharding). Shadeform box example:
`shadeform@<ip>`, key `~/.ssh/grads_bros`, passwordless `sudo -i` → root.

### 1. Provision (the `setup-trainer` skill, adapted)
Run `bootstrap.sh` **as root** (`sudo -i`), `NO_LAUNCH=1` (so it doesn't auto-launch the validator
docker-compose). It wires the nvidia container runtime, venv, go-task, node. If the driver +
`nvidia-ctk` are already present it won't reboot. No `/ephemeral` on most B300 boxes → Docker stays on
`/` (plenty of room). Adapt the skill's `root@` to `shadeform@` + `sudo`.

### 2. Build the Blackwell image
`docker build -f dockerfiles/standalone-text-trainer.dockerfile -t quasar-trainer:b300 .`
(cu128 torch, triton 3.7.1, CUDA-12.9 ptxas, no flash-attn — all baked). ~10–15 min (no FA compile).

### 3. Get the model (weights + isolating packing build)
- Pristine weights + raven: `huggingface-cli download silx-ai/Quasar-Preview` + `eyad-silx/raven`
  (raven/** into the model dir). `trainer_downloader.py` does both.
- Overlay the **isolating** modeling code from `wanderingweights/quasar-packing` (or the box's
  `quasar-edited/`): the 3 edited `.py` (`modeling_quasar_long.py`, `raven/layers/raven.py`,
  `fla/layers/quasar.py`). Easiest: `cp -al` the pristine dir, then break-hardlink + overwrite those 3
  files (leaves pristine untouched). Mount the combined dir to
  `/cache/models/silx-ai--Quasar-Preview`.

### 4. Stage data
Test split (smoke): the stage-1 presigned B2 URL → save as
`<datasets>/<task_id>_train_data.json` (ChatTask conversations format). Full data: the
sft-e2e-pipeline output. (`.runenv`/B2 creds are box-only, never in the repo.)

### 5. Launch (the verified command)
```bash
docker run -d --name quasar_run --gpus all --shm-size=32g \
  -v <model_dir>:/cache/models/silx-ai--Quasar-Preview \
  -v <datasets_dir>:/cache/datasets \
  -v <ckpt_dir>:/app/checkpoints \
  -v <triton_cache>:/root/.triton \
  -e SN56_VERBOSE=1 \
  -e HUGGINGFACE_TOKEN=<gradients token; box-only> \
  -e HF_CKPT_REPO=<org>/<repo>           # enables checkpoint -> HF
  -e WANDB_API_KEY=<key> -e WANDB_MODE=online -e WANDB_PROJECT=quasar-sft-e2e -e WANDB_ENTITY=subiawaud \
  quasar-trainer:b300t \                 # b300t = b300 + triton-3.7.1 overlay
  --task-id <id> --model silx-ai/Quasar-Preview \
  --dataset /cache/datasets/<id>_train_data.json \
  --dataset-type '{"chat_template":"chatml","chat_column":"conversations","chat_role_field":"from","chat_content_field":"value","chat_user_reference":"user","chat_assistant_reference":"assistant"}' \
  --task-type ChatTask --file-format json \
  --hours-to-complete 24 --expected-repo-name <run_name>
```

### 6. Quasar-specific behavior (all gated on `is_quasar` in `train_instruct.py`)
- `packing=True`, `packing_mode="naive"`, buffer `max_length=4096`, `per_device_train_batch_size=4`
  (bs=8 OOMs at 32768 tok/microbatch).
- `_tied_weights_keys` list→None (the save fix).
- LR search uses the 8-bit optimizer (the OOM fix). **Note:** the `quasar-ddp` branch currently has an
  `elif is_quasar:` that **bypasses** the search and forces `lr=1.6e-4` (its consistent S1 best) for
  fast iteration — **remove that branch to re-enable the live finder** for a real run.
- `_HFCkptUpload` pushes checkpoints to `HF_CKPT_REPO`.
- `SAVE_SMOKE=1` env → accum=1 + frequent saves (verification only).

### 7. Gotchas / knobs
- **bs ceiling**: bs=4 × 4096 ≈ 267 GB (near the 275 GB limit). bs=8 OOMs. Don't raise without freeing
  memory (multi-GPU / optimizer offload).
- **Saves are end-time/eval-driven**, gated to skip evals before 0.75 epochs. With low `--hours`, the
  **end-time save** fires fast (it's never gated). `periodic_save_steps` does *not* override that gate.
- **Giants > buffer (full data, ~15k tok)**: TODO. A sample longer than the 4096 buffer overflows
  `pack_data_points_naive`'s fixed-len assert. Needs an own-buffer/bs=1 path (token-budget batching) —
  don't raise `max_length` past the data's max without it.
- **chatml vs native template**: the `chatml` dataset-type warns `EOS <|endoftext|> not in template`
  and starts loss ~3.5 (vs ~1.5 native). Switch `--dataset-type` to the model's own `chat_template`
  for a faithful run.
- Don't `docker rm -f` a GPU-busy container repeatedly — it can wedge the nvidia runtime.

### 8. Verify
- Packing: `[sn56][pad]` logs show `packed=True` + ~88% real.
- Throughput: `[sn56][tput]` logs real tok/s per step.
- Save→HF: `[sn56][hf] uploaded checkpoint-N -> <repo>`.
- wandb: https://wandb.ai/subiawaud/quasar-sft-e2e
