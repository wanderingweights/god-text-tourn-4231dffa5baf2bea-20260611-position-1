import quiet_mode  # noqa: F401,E402 — competition log gate; must precede heavy imports
from typing import Dict, Optional
import requests
import json
import random
import utility
from datasets import Dataset
from utility import log_info
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
import transformers
import torch
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from transformers import Trainer
from trl import DPOTrainer, DPOConfig, ModelConfig, ScriptArguments, TrlParser
from trl import get_kbit_device_map, get_peft_config, get_quantization_config
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModelForCausalLM,
    AutoPeftModelForCausalLM,
)
from transformers import TrainerCallback
import argparse
import os
from customized_trainer import resize_if_needed, set_generation_config, CustomEvalSaveCallback, WhenToEvalHandler, init_wandb
from checkpoint_avg_callback import AdaptiveTrainingCallback
from state_manager import get_state, set_state

# from packing.packed_dataset import PackedDataset
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
import datetime
import shutil
from huggingface_hub import HfApi
from typing import Callable, Optional
import bitsandbytes as bnb
import yaml
from tokenize_dpo import get_dataset
from transformers.modeling_utils import is_deepspeed_zero3_enabled



LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


@dataclass
class TrainingArguments(DPOConfig):
    request_path: Optional[str] = field(default=None)
    use_liger: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)
    use_attn_implementation: Optional[str] = field(default="")


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    lora_param_count = 0
    all_param = 0
    embedding_lm_head_param_count = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            log_info(f"ajustável: {name}, pesos={num_params}")
            if "lm_head" in name or "embed_tokens" in name:
                embedding_lm_head_param_count += num_params
            else:
                lora_param_count += num_params
    trainable_params = embedding_lm_head_param_count + lora_param_count
    log_info(
        f"total={all_param:,d} || ajustáveis={trainable_params:,d} || fração={100 * trainable_params / all_param:.2f}%"
    )
    log_info(
        f"cabeça_emb={embedding_lm_head_param_count} = {embedding_lm_head_param_count * 100 / all_param:.2f}%"
    )
    log_info(
        f"lora={lora_param_count} = {lora_param_count * 100 / all_param:.2f}%"
    )


def get_max_length_config():
    config_path = "test_axolotl.yml"
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict["sequence_len"]


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (TrainingArguments, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "dpo", help="Run the DPO training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


def main():
    """Format of training requests"""
    parser = make_parser()
    training_args, model_args = parser.parse_args_and_config()
    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]

    # check if need to run early stop or not
    task_id = train_request["task_id"]
    
    # wandb_init_success = init_wandb(train_request)
    # if not wandb_init_success:
    #     log_info("WANDB_API_KEY is not set, do not report to wandb")
    #     training_args.report_to = "none"    
    # else:
    #     log_info("WANDB_API_KEY is provided, we will report to wandb")
    #     training_args.report_to = "wandb"

    # log_info(f"Training request: {train_request}", "start")
    # first download the dataset from the URL, save it as data.json
    output_dir = training_args.output_dir
    from model_utility import load_tokenizer
    tokenizer = load_tokenizer(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # max_length = get_max_length_config()
    # if "max_length" in train_request:
    #     max_length = train_request["max_length"]
    # default implementation, max_length=1024 (prompt + completion), max_prompt_length=512

    train_path = os.path.join("datasets", f"dpo_train_{task_id}.json")
    dev_path = os.path.join("datasets", f"dpo_dev_{task_id}.json")

    train_ds = get_dataset(train_path, train_request["dataset_type"])
    dev_ds = get_dataset(dev_path, train_request["dataset_type"])

    log_info(f"nós={training_args.world_size}")
    total_steps_per_epoch = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )

    log_info(f"passos/época={total_steps_per_epoch}")
    # consider reducing the batch_size if it is quite big
    # num_steps = len(train_ds) * training_args.num_train_epochs / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)
    # num_steps > min_step ->
    max_batch_size_theory = len(train_ds) / (
        training_args.gradient_accumulation_steps
        * training_args.world_size
        * train_request["min_steps"]
    )
    max_batch_size_theory = int(max_batch_size_theory)
    if max_batch_size_theory == 0:
        max_batch_size_theory = 1

    original_batch_size = training_args.per_device_train_batch_size

    quantization_config = get_quantization_config(model_args)
    device_string = "cuda:" + str(LOCAL_RANK)
    
    device_map=(
            get_kbit_device_map()
            if quantization_config is not None
            else {"": device_string}
        )
    if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
        device_map = None


    attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager"
    if training_args.use_attn_implementation:
        attn_implementation = training_args.use_attn_implementation
        log_info(f"atenção via {attn_implementation}")
        
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map
    )
    
    # Only add quantization_config if it's not None
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    log_info(f"args finais: {training_args}")

    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(train_request["model_path"], **model_kwargs)
    if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
        # set gradient checkpointing to True with use_reentrant=True for deepspeed
        log_info("checkpointing ativado com reentrada para deepspeed")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})


    # some model need to set the generation config or encounter the invalid generation config error
    set_generation_config(train_request["model_name"], model)

    ref_model = None
    if "ref_model" in train_request:
        ref_model = model_class.from_pretrained(
            train_request["ref_model"], **model_kwargs
        )
        # print("load ref_model: ", train_request["ref_model"])

    peft_config = get_peft_config(model_args)
    if "lora_model" in train_request:
        model = PeftModelForCausalLM.from_pretrained(
            model, train_request["lora_model"], is_trainable=True, **model_kwargs
        )

    if peft_config is None:  # this is full-weight training
        # some model need to resize the token embeddings or encounter the size mismatch error; only for full-weight models
        resize_if_needed(train_request["model_name"], model, len(tokenizer))

    # Only resize token embeddings if not using LoRA
    # if peft_config is None:  # full-weights training
    #    model.resize_token_embeddings(len(tokenizer))

    # Check if this is the main process and create the output directory
    if is_main_process(LOCAL_RANK):  # Only create directory on main process
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"destino criado: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    log_info(f"salvo periódico a cada {periodic_save_steps} passos")

    training_args.save_only_model = True  # only save the model, not the optimizer

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    log_info(f"colunas: {train_ds.column_names}")

    max_steps = train_request.get("max_steps", -1)
    log_info(f"teto de passos={max_steps}")
    
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = get_state()
    if "train" not in state:
        state["train"] = {}
    state["train"]["start_train_time"] = start_time
    if is_main_process(LOCAL_RANK):
        set_state(state)
    
    total_steps_per_epoch = len(train_ds) // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
    
    total_steps_all_epochs = total_steps_per_epoch * training_args.num_train_epochs
    log_info(f"passos/época={total_steps_per_epoch}, total={total_steps_all_epochs}")

    # Refine warmup: 3% of total steps, capped by the hours-based value from config
    warmup_from_ratio = max(10, int(0.03 * total_steps_all_epochs))
    training_args.warmup_steps = min(training_args.warmup_steps, warmup_from_ratio)
    log_info(f"Alongando por {training_args.warmup_steps} passos antes da corrida")

    # Adaptive eval frequency (matches instruct pipeline)
    _cm = train_request.get("checking_mode")
    is_final_run = _cm in ("none", None)
    log_info(f"modo={_cm!r}, final={is_final_run}")
    if is_final_run:
        hours = train_request.get("hours_to_complete", 2)
        max_evals = max(3, int(hours * 4))
        adaptive_eval_steps = max(30, total_steps_all_epochs // min(10, max_evals))
        training_args.eval_strategy = "steps"
        training_args.eval_steps = adaptive_eval_steps
        training_args.save_strategy = "steps"
        training_args.save_steps = adaptive_eval_steps
        log_info(f"Olhando no espelho a cada {adaptive_eval_steps} passos")

    success_file = os.path.join(training_args.output_dir, "success.txt")
    # remove the success file if it exists
    if is_main_process(LOCAL_RANK) and os.path.exists(success_file):
        os.remove(success_file)
    
    checking_step = train_request["checking_step"]
    if checking_step >= total_steps_per_epoch:
        checking_step = total_steps_per_epoch - 2

    # ── Always-on NEFTune ──
    _neft_alpha = 1
    baseline_stats = train_request.get("baseline_stats")
    if baseline_stats:
        grad_noise = baseline_stats.get("training", {}).get("gradient_noise_scale", 0.0)
        if grad_noise > 1.0:
            _neft_alpha = 5
            log_info(f"[sn56][neftune] grad_noise={grad_noise:.2f} > 1.0, alpha=5")
    training_args.neftune_noise_alpha = _neft_alpha
    log_info(f"[sn56][neftune] alpha={_neft_alpha}")

    # Averaging mode: in-RAM window for non-sharded models that fit; disk
    # averaging for sharded / too-big (in-RAM snapshots are invalid sharded).
    _shard_ds = getattr(training_args, "deepspeed", None) is not None
    _sharded = _shard_ds or len(training_args.fsdp) > 0
    _trainable_bytes = sum(p.numel() for p in model.parameters() if p.requires_grad) * 2
    _avg_mode = "disk" if (_sharded or 6 * _trainable_bytes > 100e9) else "ram"
    log_info(f"[sn56][caldo] modo={_avg_mode} (treináveis={_trainable_bytes / 1e9:.0f}GB, sharded={_sharded})")
    ckpt_avg = (
        AdaptiveTrainingCallback(
            window=3, averaging_mode=_avg_mode, output_dir=training_args.output_dir
        )
        if is_final_run else None
    )
    if ckpt_avg is not None:
        ckpt_avg._submission_dir = train_request["submission_dir"]

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            *([] if ckpt_avg is None else [ckpt_avg]),
            CustomEvalSaveCallback(
                WhenToEvalHandler(train_request["end_time"], train_request["save_before_remaining_time"], periodic_save_steps=periodic_save_steps, steps_per_epoch=total_steps_per_epoch, max_steps=max_steps),
                train_request["submission_dir"],
                training_args.output_dir,
                train_request["model_name"],
                max_steps,
                checking_step=checking_step,
                total_steps_all_epochs=total_steps_all_epochs,
                end_time=train_request["end_time"],
                checking_mode=train_request.get("checking_mode", "none"),
                use_reward_accuracy=False,
            )
        ],
    )

    if ckpt_avg is not None:
        ckpt_avg.trainer = trainer

    # DPO/GRPO lr_search: use trainer.training_step instead of raw model(**batch).
    # DPO collator produces paired chosen/rejected batches that only the DPOTrainer
    # knows how to process. We use the Trainer's own training_step for each trial.
    #
    # IMPORTANT: DPOTrainer.get_batch_loss_metrics() calls
    # accelerator.gather_for_metrics() 8+ times per training_step — these are
    # NCCL all_gather collectives. The lr_search loop terminates based on LOCAL
    # losses (rank-independent), so ranks may run different numbers of trials.
    # Different trial counts → mismatched collective calls → NCCL deadlock.
    # Fix: disable gather_for_metrics during lr_search so each rank runs
    # independently, then sync results via explicit broadcast afterward.
    _use_deepspeed = getattr(training_args, "deepspeed", None) is not None
    # Skip LR search if a previous attempt already found the best LR.
    # Scale for batch size changes: lr ∝ sqrt(bs) (square root scaling rule).
    _state_cache = get_state()
    _cached_lr = _state_cache.get("best_lr_found")
    _cached_bs = _state_cache.get("best_lr_batch_size")
    if _cached_lr is not None:
        import math
        _current_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        if _cached_bs is not None and _cached_bs != _current_bs:
            _scale = math.sqrt(_current_bs / _cached_bs)
            _adjusted_lr = _cached_lr * _scale
            log_info(f"[sn56][farejando] LR cached {_cached_lr:.2e} ajustado pra bs {_cached_bs}->{_current_bs}: {_adjusted_lr:.2e}")
            _cached_lr = _adjusted_lr
        training_args.learning_rate = _cached_lr
        log_info(f"[sn56][farejando] Usando LR cached de tentativa anterior: {_cached_lr:.2e}")
    elif _use_deepspeed:
        log_info(f"[sn56][farejando] Pulando (DeepSpeed ativo)")
    else:
        from lr_search import run_lr_search
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{LOCAL_RANK}")
            model = model.to(device)
            log_info(f"[sn56][farejando] Modelo movido pra {device}")

        search_loader = trainer.get_train_dataloader()

        def _halve_batch_dataloader():
            """Rebuild dataloader with half the current batch size (OOM fallback)."""
            cur = training_args.per_device_train_batch_size
            new_bs = max(1, cur // 2)
            training_args.per_device_train_batch_size = new_bs
            trainer._train_batch_size = training_args.train_batch_size
            log_info(f"[sn56][farejando] Cozinha quente, lote {cur} -> {new_bs}")
            return trainer.get_train_dataloader()

        # Disable NCCL collectives inside DPO metrics so each rank can run
        # lr_search independently (different trial counts are fine).
        _orig_gather = trainer.accelerator.gather_for_metrics
        trainer.accelerator.gather_for_metrics = lambda tensors, *a, **kw: tensors
        original_bs = training_args.per_device_train_batch_size
        # initial_lr is now a DPO-scale estimate (lr_finder DpoTask correction +
        # capped at the dpo_config bucket); the held-out-scored sweep + extend-low
        # refine around it. No extra scaling here — that would double-correct.
        try:
            best_lr, _t_per_step = run_lr_search(
                model=model,
                train_dataloader=search_loader,
                initial_lr=training_args.learning_rate,
                hours_to_complete=train_request["hours_to_complete"],
                grad_accum_steps=training_args.gradient_accumulation_steps,
                max_grad_norm=training_args.max_grad_norm,
                dataloader_factory=_halve_batch_dataloader,
                trainer=trainer,
                steps_per_epoch=total_steps_per_epoch,
                eval_dataloader=trainer.get_eval_dataloader(),
            )
        finally:
            trainer.accelerator.gather_for_metrics = _orig_gather
            # Clear stale metrics accumulated during lr_search trials
            from collections import defaultdict
            trainer._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Sync batch size across ranks: if ANY rank halved its batch on OOM
        # during the search, every rank must adopt the smallest surviving batch.
        # Otherwise divergent per-rank steps-per-epoch deadlock DDP at an epoch
        # boundary. all_reduce(MIN) is a collective — all ranks must call it.
        if torch.distributed.is_initialized():
            _bs_t = torch.tensor(
                [training_args.per_device_train_batch_size],
                device=next(model.parameters()).device,
            )
            torch.distributed.all_reduce(_bs_t, op=torch.distributed.ReduceOp.MIN)
            _min_bs = int(_bs_t.item())
            if _min_bs < training_args.per_device_train_batch_size:
                log_info(
                    f"[sn56][farejando] OOM em outro rank: lote "
                    f"{training_args.per_device_train_batch_size} -> {_min_bs} (sincronizado)"
                )
                training_args.per_device_train_batch_size = _min_bs

        if training_args.per_device_train_batch_size != original_bs:
            trainer._train_batch_size = training_args.train_batch_size
            log_info(f"[sn56][farejando] Lote reduzido durante busca: {original_bs} -> {training_args.per_device_train_batch_size} (carries to training)")

        if torch.distributed.is_initialized():
            lr_tensor = torch.tensor([best_lr], device=next(model.parameters()).device)
            torch.distributed.broadcast(lr_tensor, src=0)
            best_lr = lr_tensor.item()
            # Sync weights — ranks may have restored different best trials
            # (local tail-avg losses differ across shards)
            for p in model.parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)
            log_info(f"[sn56][farejando] Sincronizando lr={best_lr:.2e} + weights to all ranks")

        trainer.args.learning_rate = best_lr
        log_info(f"[sn56][farejando] Usando lr={best_lr:.2e}")
        # Cache LR + batch size so OOM retries don't repeat the search
        _s = get_state()
        _s["best_lr_found"] = best_lr
        _s["best_lr_batch_size"] = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        if is_main_process(LOCAL_RANK):
            set_state(_s)

        # ── Time-aware epoch planning (same as instruct) ──
        # Compute on rank 0, broadcast to all ranks to avoid NCCL deadlock
        # from mismatched eval_steps.
        if _t_per_step is not None:
            _epoch_info = torch.tensor([0.0, 0.0, 0.0], device=next(model.parameters()).device)

            if is_main_process(LOCAL_RANK):
                import datetime as _dt_mod
                _now = _dt_mod.datetime.now(_dt_mod.timezone.utc)
                _end_dt = _dt_mod.datetime.strptime(
                    train_request["end_time"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=_dt_mod.timezone.utc)
                _remaining = (_end_dt - _now).total_seconds() * 0.85
                _achievable = int(_remaining / _t_per_step)
                _spe = len(train_ds) // max(1,
                    training_args.per_device_train_batch_size
                    * training_args.gradient_accumulation_steps
                    * training_args.world_size
                )
                if _spe > 0:
                    _max_ep = 4.0 if len(train_ds) < 10000 else 3.0
                    # +25%: _t_per_step is now the REAL post-search per-step time, so
                    # _achievable is accurate (only the 0.85 buffer is conservative).
                    # 1.25 just spends that buffer — plans slightly past the deadline
                    # to use the full budget while keeping cosine ~94% cooled at the
                    # cut. save_before_remaining_time is the real stop; floor 1.3.
                    _opt_epochs = round(max(1.3, min(_max_ep, 1.25 * _achievable / _spe)), 2)
                    _new_total = int(_spe * _opt_epochs)
                    _warmup = min(training_args.warmup_steps, max(10, int(0.03 * _new_total)))
                    _eval_s = float(training_args.eval_steps)
                    if is_final_run:
                        # Eval every 1/5 epoch — matches instruct; pairs with
                        # the 0.75-epoch skip in CustomEvalSaveCallback.
                        _eval_s = float(max(30, _spe // 5))
                    _epoch_info[0] = _opt_epochs
                    _epoch_info[1] = float(_warmup)
                    _epoch_info[2] = _eval_s
                    log_info(
                        f"[sn56][relogio] Epochs {training_args.num_train_epochs} -> {_opt_epochs} "
                        f"(t/step={_t_per_step:.3f}s, steps/ep={_spe}, achievable={_achievable})"
                    )

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(_epoch_info, src=0)

            if _epoch_info[0] > 0:
                training_args.num_train_epochs = _epoch_info[0].item()
                training_args.warmup_steps = int(_epoch_info[1].item())
                if is_final_run:
                    training_args.eval_steps = int(_epoch_info[2].item())
                    training_args.save_steps = int(_epoch_info[2].item())

    log_info("Bora treinar")
    log_info(f"fiscalização a cada {training_args.eval_steps} passos, {len(dev_ds)} na prova")
    trainer.train()

    # ── Final dev-data pass (data maximization) ──
    # Same as instruct: from the best checkpoint, one low-LR epoch over the
    # held-out dev set (sharded DDP, training's effective batch), then atomic
    # weights-only save. run_dev_pass uses trainer.training_step, so the loss is
    # the proper DPO chosen/rejected + reference-model loss. Best-effort + gated.
    try:
        _dev_n = len(dev_ds)
        if is_final_run and _avg_mode == "ram" and _dev_n > 0:
            from dev_pass import run_dev_pass
            _min_lr_rate = 0.25
            _kw = training_args.lr_scheduler_kwargs or {}
            if isinstance(_kw, str):
                try:
                    _kw = json.loads(_kw)
                except (json.JSONDecodeError, ValueError):
                    _kw = {}
            try:
                _min_lr_rate = float(_kw.get("min_lr_rate", 0.25))
            except (AttributeError, TypeError, ValueError):
                pass
            _min_lr = trainer.args.learning_rate * _min_lr_rate
            log_info(f"[sn56][devfit] {_dev_n} amostras de dev, 1 época @ lr={_min_lr:.2e}")
            run_dev_pass(
                trainer,
                submission_dir=train_request["submission_dir"],
                min_lr=_min_lr,
                max_grad_norm=training_args.max_grad_norm,
                train_per_device=training_args.per_device_train_batch_size,
                train_grad_accum=training_args.gradient_accumulation_steps,
                local_rank=LOCAL_RANK, log=log_info,
            )
        else:
            log_info(f"[sn56][devfit] pulando (final={is_final_run}, modo={_avg_mode}, dev={_dev_n})")
    except Exception as _e:
        log_info(f"[sn56][devfit] dev-pass falhou ({_e}), mantendo melhor checkpoint")

    if is_main_process(LOCAL_RANK):
        with open(os.path.join(training_args.output_dir, "success.txt"), "w") as f:
            f.write("Success")


if __name__ == "__main__":
    main()