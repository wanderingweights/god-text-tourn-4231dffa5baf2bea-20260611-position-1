import quiet_mode  # noqa: F401,E402 — competition log gate; must precede heavy imports
from typing import Dict, Optional
import requests
import json
import random
from utility import log_info, MyDataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from transformers import Trainer
from customized_trainer import resize_if_needed, set_generation_config, CustomEvalSaveCallback, WhenToEvalHandler, init_wandb
from checkpoint_avg_callback import AdaptiveTrainingCallback
from lr_search import run_lr_search
from kl_trainer import KLRegularizedTrainer, load_base_model

# from packing.packed_dataset import PackedDataset
from transformers import (
    Trainer,
    TrainingArguments,
)

import os
import datetime
import shutil
from huggingface_hub import HfApi
from typing import Callable, Optional
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml
from state_manager import get_state, set_state

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    request_path: Optional[str] = field(default=None)
    packing: Optional[bool] = field(default=False)
    max_packed_size: Optional[int] = field(default=-1)
    use_liger: Optional[bool] = field(default=False)
    use_lora: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)
    use_attn_implementation: Optional[str] = field(default="")

@dataclass
class LoraArguments:
    lora_r: int = 128
    lora_alpha: int = 512
    lora_dropout: float = 0.1
    lora_target_modules: str = "all"  # all for all linear; "q_proj v_proj"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    
    
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
        f"total={all_param:,d} || ajustáveis={trainable_params:,d} || fração={100 * trainable_params / all_param:.1f}%"
    )
    log_info(
        f"cabeça_emb={embedding_lm_head_param_count} ({embedding_lm_head_param_count * 100 / all_param:.1f}%)"
    )
    log_info(
        f"lora={lora_param_count} ({lora_param_count * 100 / all_param:.1f}%)"
    )
    

def load_lora_model(training_args: TrainingArguments, model_path: str, lora_args: LoraArguments, token_nums: int):
    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
        torch_dtype=torch.bfloat16,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if lora_args.q_lora
            else None
        ),
    )
    # do not resize tokem embeddings in LOra --> will encounter size mismatch error in evaluation 
    # model.resize_token_embeddings(token_nums)
    # convert to lora
    if lora_args.lora_target_modules == "all":
        target_modules = find_all_linear_names(model)
    else:
        modules = lora_args.lora_target_modules.split(" ")
        target_modules = [mod.strip() for mod in modules if len(mod.strip()) > 0]

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
        # modules_to_save=["lm_head", "embed_tokens"],  # because we retrain the embedding
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model.config.use_cache = False
    # Activate computing load balancing loss iin MixtralForCausalLM
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)

    print_trainable_parameters(model)
    return model


def load_model(training_args: TrainingArguments, model_path: str, token_nums: int):
    model_class = transformers.AutoModelForCausalLM
    
    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        log_info("--- turbo ativo ---")
        model_class = AutoLigerKernelForCausalLM
    
    attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager"
    if training_args.use_attn_implementation:
        attn_implementation = training_args.use_attn_implementation
        log_info(f"atenção via {attn_implementation}")
    log_info(f"atenção: {attn_implementation}")
    
    model = model_class.from_pretrained(
        model_path,
        # trust_remote_code=True, remove this because we already filter the model architecture, it will not be used with liger-kernel 
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    # model.resize_token_embeddings(token_nums)
    return model


def get_max_length_config():
    config_path = "test_axolotl.yml"
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict["sequence_len"]


def main():
    """Format of training requests"""
    argument_parser = transformers.HfArgumentParser((TrainingArguments, LoraArguments))
    (training_args, lora_args) = argument_parser.parse_args_into_dataclasses()
    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    # log_info(f"Training request: {train_request}", "start")
    task_id = train_request["task_id"]

    # ── KL-regularised task contract (G.O.D feature/instruct-kl-training) ──
    # KL tasks set USE_KL=1 / KL_COEF=<float> on the container. The validator
    # scores eval_loss + kl_coef * KL(finetuned || base) over completion tokens,
    # so we add the identical term to the training loss (see kl_trainer.py).
    # Absent/unset => no KL, identical behaviour to before.
    use_kl = os.environ.get("USE_KL") == "1"
    _kl_coef_env = os.environ.get("KL_COEF")
    kl_coef = 0.0
    if use_kl and _kl_coef_env:
        try:
            kl_coef = float(_kl_coef_env)
        except (ValueError, TypeError):
            log_info(f"[sn56][kl] KL_COEF inválido ({_kl_coef_env!r}), desligando KL")
            kl_coef = 0.0
    use_kl = use_kl and kl_coef > 0
    if use_kl:
        log_info(f"[sn56][kl] USE_KL=1, kl_coef={kl_coef} — termo KL ativo")

    from model_utility import load_tokenizer
    tokenizer = load_tokenizer(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # wandb_init_success = init_wandb(train_request)
    # if not wandb_init_success:
    #     log_info("WANDB_API_KEY is not set, do not report to wandb")
    #     training_args.report_to = "none"    
    # else:
    #     log_info("WANDB_API_KEY is provided, we will report to wandb")
    #     training_args.report_to = "wandb"
        
    max_length = get_max_length_config()
    if "max_length" in train_request:
        max_length = train_request["max_length"]

    # we already tokenize the data and save it to .pt (torch format, fast)
    train_ds = MyDataset(
        tokenizer,
        f"datasets/train_tokenized_{task_id}.json",
        max_length
    )

    dev_ds = MyDataset(
        tokenizer,
        f"datasets/dev_tokenized_{task_id}.json",
        max_length
    )
    log_info(f"treino={len(train_ds)}, teste={len(dev_ds)}")

    # ── Training data deduplication ──
    # Removes exact-duplicate tokenized samples. Especially impactful for
    # datasets with high near-duplicate rates (e.g. SciEntsBank at 59%).
    baseline_stats = train_request.get("baseline_stats")
    near_dup_rate = 0.0
    if baseline_stats:
        near_dup_rate = baseline_stats.get("dataset", {}).get("near_duplicate_rate", 0.0)
    if near_dup_rate > 0.2:
        from data_filter import deduplicate_samples
        before = len(train_ds.eval_dataset)
        train_ds.eval_dataset = deduplicate_samples(train_ds.eval_dataset)
        log_info(f"[sn56][dedup] near_dup_rate={near_dup_rate:.2f}, deduped {before} -> {len(train_ds.eval_dataset)}")

    # ── (d) Prompt loss weight for prompt-dominated datasets ──
    # When prompt:completion > 5:1, unmask a small fraction of prompt tokens
    # so the model gets gradient signal about the input language/domain.
    # PLW scales inversely with ratio to avoid gradient dilution on extreme
    # ratios (e.g. 28:1 at flat 5% = 33% gradient dilution).
    if baseline_stats:
        _ds = baseline_stats.get("dataset", {})
        _prompt_tok = _ds.get("prompt_tokens", 0)
        _comp_tok = _ds.get("completion_tokens", 1)
        _pc_ratio = _prompt_tok / max(_comp_tok, 1)
        if _pc_ratio > 5:
            _plw = 0.05 / max(1.0, _pc_ratio / 5.0)
            from utility import apply_prompt_loss_weight
            train_ds.eval_dataset = apply_prompt_loss_weight(train_ds.eval_dataset, plw=_plw)
            log_info(f"[sn56][plw] prompt:comp={_pc_ratio:.1f}:1, PLW={_plw:.4f}")

    donot_pack = False
    original_train_size = len(train_ds)
    original_steps = original_train_size // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )  # number of steps in the original training
    # min_steps here is per epoch
    if original_steps < train_request["min_steps"]:
        donot_pack = True
        log_info(f"passos={original_steps} < mínimo={train_request['min_steps']}, sem compactação")

    min_data_size_num = (
        train_request["min_steps"]
        * training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    
        
    log_info(f"limiar={min_data_size_num}, janela={max_length}")
    packing_mode = train_request.get("packing_mode", "fa")
    use_fa_packing = packing_mode == "fa"
    # Keep a handle on the RAW (unpacked) dataset: the post-search Gaussian
    # subsample mutates its eval_dataset and re-packs — PackedDataset doesn't
    # expose the underlying samples, so without this reference the subsample
    # can never fire on the packed (default) path.
    _raw_train_ds = train_ds
    _train_is_packed = False
    if training_args.packing and not donot_pack:
        from monkeypatch import monkey_patch_packing_for_model, PackedDataset
        if use_fa_packing:
            log_info("costurando sequências com atenção rápida")
            monkey_patch_packing_for_model(train_request["model_path"])
        else:
            log_info("costura ingênua, posições reiniciadas por sequência")

        t1 = datetime.datetime.now()
        train_ds = PackedDataset(
            train_ds,
            tokenizer,
            max_input_length=max_length,
            max_packed_size=training_args.max_packed_size,
            min_item_num=min_data_size_num,
            use_fa=use_fa_packing,
        )
        _train_is_packed = True
        t2 = datetime.datetime.now()
        log_info(f"costura treino: {(t2 - t1).total_seconds()}s")
        # Only pack eval with FA (proper sequence isolation).
        # Naive packing allows cross-attention between sequences which
        # corrupts eval loss — use unpacked eval for reliable checkpoint selection.
        if use_fa_packing:
            t1 = datetime.datetime.now()
            dev_ds = PackedDataset(
                dev_ds,
                tokenizer,
                max_input_length=max_length,
                max_packed_size=training_args.max_packed_size,
                use_fa=True,
            )
            t2 = datetime.datetime.now()
            log_info(f"costura teste: {(t2 - t1).total_seconds()}s")
        else:
            log_info("teste sem costura — isolamento necessário")
        log_info(f"treino compactado: {train_ds.stat()}")
        if hasattr(dev_ds, 'stat'):
            log_info(f"teste compactado: {dev_ds.stat()}")
        else:
            log_info(f"teste solto: {len(dev_ds)} amostras")

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
    if training_args.per_device_train_batch_size > max_batch_size_theory:
        # if batch_size is quite big set it to this value to make sure that we have at least min_steps
        if train_request.get("adjust_batch_size", True):
            log_info(
                f"lote grande demais ({training_args.per_device_train_batch_size}), cortando para {max_batch_size_theory}"
            )
            training_args.per_device_train_batch_size = max_batch_size_theory
            # need to update total_steps_per_epoch
            total_steps_per_epoch = len(train_ds) // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
            log_info(f"passos/época recalculados={total_steps_per_epoch}")

    if training_args.use_lora:
        model = load_lora_model(training_args, train_request["model_path"], lora_args, len(tokenizer))
    else:
        model = load_model(training_args, train_request["model_path"], len(tokenizer))
        # some model need to resize the token embeddings or encounter the size mismatch error; only for full-weight models
        resize_if_needed(train_request["model_name"], model, len(tokenizer))
    
    try:
        model.config.use_cache = False
    except:
        pass
    
    # some model need to set the generation config or encounter the invalid generation config error
    set_generation_config(train_request["model_name"], model)

    # Check if this is the main process and create the output directory
    if is_main_process(LOCAL_RANK):  # Only create directory on main process
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"destino criado: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    log_info(f"salvo periódico a cada {periodic_save_steps} passos")
    training_args.save_only_model = True  # only save the model, not the optimizer
    
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

    warmup_from_ratio = max(10, int(0.03 * total_steps_all_epochs))
    training_args.warmup_steps = min(training_args.warmup_steps, warmup_from_ratio)
    log_info(f"Alongando por {training_args.warmup_steps} passos antes da corrida")

    # Adaptive eval frequency: only on final run (not LR search probes)
    _cm = train_request.get("checking_mode")
    is_final_run = _cm in ("none", None)
    log_info(f"checking_mode={_cm!r}, is_final_run={is_final_run}")
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
    # Start with alpha=1 (barely noticeable) so the hook is active.
    # Rollback can escalate to 5 → 10 → 15 when overfitting is detected.
    # Higher initial alpha for noisy datasets where regularization is needed from step 1.
    _neft_alpha = 1
    if baseline_stats:
        grad_noise = baseline_stats.get("training", {}).get("gradient_noise_scale", 0.0)
        if grad_noise > 1.0:
            _neft_alpha = 5
            log_info(f"[sn56][neftune] grad_noise={grad_noise:.2f} > 1.0, alpha=5")
    training_args.neftune_noise_alpha = _neft_alpha
    log_info(f"[sn56][neftune] alpha={_neft_alpha}")

    # Averaging mode: in-RAM window averaging for non-sharded models that fit
    # host RAM; disk-based averaging of consolidated checkpoints for sharded
    # (FSDP/DeepSpeed) or too-big models, where in-RAM snapshots are only shards.
    _shard_ds = getattr(training_args, "deepspeed", None) is not None
    _sharded = _shard_ds or len(training_args.fsdp) > 0
    _trainable_bytes = sum(p.numel() for p in model.parameters() if p.requires_grad) * 2
    # RAM peak holds ~6x trainable bytes (best + 3-window + avg + stash).
    _avg_mode = "disk" if (_sharded or 6 * _trainable_bytes > 100e9) else "ram"
    log_info(
        f"[sn56][caldo] modo={_avg_mode} (treináveis={_trainable_bytes / 1e9:.0f}GB, "
        f"sharded={_sharded})"
    )
    ckpt_avg = (
        AdaptiveTrainingCallback(
            window=3, averaging_mode=_avg_mode, output_dir=training_args.output_dir
        )
        if is_final_run else None
    )
    if ckpt_avg is not None:
        ckpt_avg._submission_dir = train_request["submission_dir"]
        # end_time bounds the greedy soup's eval time at train end (it may use
        # at most half the remaining wall-clock; the dev-pass uses the rest).
        ckpt_avg.end_time = train_request["end_time"]

    # KL tasks use the KL-regularised subclass (only compute_loss differs, so all
    # callbacks/averaging/dev-pass/wall-clock behave identically); the base-model
    # reference is wired in after the LR search to keep it out of OOM-sensitive
    # search memory. Non-KL tasks use the stock Trainer unchanged.
    _trainer_cls = KLRegularizedTrainer if use_kl else Trainer
    # Kept as a variable: the dev-pass budgeting below moves its end-time
    # trigger earlier once t_per_step is known (the trigger now also STOPS
    # training — see CustomEvalSaveCallback — which is what gives the dev-pass
    # its window to run in).
    when_to_eval_handler = WhenToEvalHandler(
        train_request["end_time"],
        train_request["save_before_remaining_time"],
        periodic_save_steps=periodic_save_steps,
        steps_per_epoch=total_steps_per_epoch,
        max_steps=max_steps,
    )
    _trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=[
            *([] if ckpt_avg is None else [ckpt_avg]),
            CustomEvalSaveCallback(
                when_to_eval_handler,
                train_request["submission_dir"],
                training_args.output_dir,
                train_request["model_name"],
                max_steps,
                checking_step=checking_step,
                total_steps_all_epochs=total_steps_all_epochs,
                end_time=train_request["end_time"],
                checking_mode=train_request.get("checking_mode", "none")
            )
        ],
    )
    if use_kl:
        _trainer_kwargs["kl_coef"] = kl_coef
        _trainer_kwargs["use_lora_base"] = bool(training_args.use_lora)
    trainer = _trainer_cls(**_trainer_kwargs)

    if ckpt_avg is not None:
        ckpt_avg.trainer = trainer
    trainer.tokenizer = tokenizer

    # ── Estimate dataset coverage (rough; refined with real timing below) ──
    # Coverage = how much of one epoch we can train in the budget. Drives
    # Gaussian difficulty subsampling, applied AFTER the search so it can use the
    # measured per-step time. The params heuristic here is only a fallback for
    # when the search is skipped (cached LR / DeepSpeed).
    _param_nums = sum(p.numel() for p in model.parameters())
    _est_step_time = 4.0 + _param_nums / 1e9 * 3.0  # rough fallback heuristic
    _hours = train_request.get("hours_to_complete", 2)
    _eff_bs = max(1, training_args.per_device_train_batch_size
                  * training_args.gradient_accumulation_steps
                  * training_args.world_size)
    _steps_per_epoch = len(train_ds) // _eff_bs
    _est_total_steps = int(_hours * 3600 / _est_step_time)
    _est_coverage = _est_total_steps / max(1, _steps_per_epoch)
    log_info(f"[sn56][cobertura] steps/epoch={_steps_per_epoch}, est_total={_est_total_steps}, coverage~{_est_coverage:.1%} (heurística)")

    # ── LR search ──
    # Always run the search (its warmup measures real per-step time and
    # plan_budget decides skip/validate/tiny/full). The coverage heuristic no
    # longer gates the search — a good LR matters regardless of how much of the
    # dataset we cover. Coverage still drives Gaussian subsampling above.
    _use_deepspeed = getattr(training_args, "deepspeed", None) is not None
    t_per_step = None

    # Skip LR search if a previous attempt already found the best LR.
    _state_cache = get_state()
    _cached_lr = _state_cache.get("best_lr_found")
    _cached_bs = _state_cache.get("best_lr_batch_size")
    if _cached_lr is not None:
        import math
        _current_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        if _cached_bs is not None and _cached_bs != _current_bs:
            _scale = math.sqrt(_current_bs / _cached_bs)
            _cached_lr = _cached_lr * _scale
            log_info(f"[sn56][farejando] LR cached ajustado pra bs {_cached_bs}->{_current_bs}: {_cached_lr:.2e}")
        training_args.learning_rate = _cached_lr
        log_info(f"[sn56][farejando] Usando LR cached de tentativa anterior: {_cached_lr:.2e}")
    elif _use_deepspeed:
        log_info(f"[sn56][farejando] Pulando (DeepSpeed ativo)")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{LOCAL_RANK}")
            model = model.to(device)
            log_info(f"[sn56][farejando] Modelo movido pra {device}")

        search_loader = trainer.get_train_dataloader()

        def _halve_batch_dataloader():
            cur = training_args.per_device_train_batch_size
            new_bs = max(1, cur // 2)
            training_args.per_device_train_batch_size = new_bs
            trainer._train_batch_size = training_args.train_batch_size
            log_info(f"[sn56][farejando] Cozinha quente, lote {cur} -> {new_bs}")
            return trainer.get_train_dataloader()

        original_bs = training_args.per_device_train_batch_size
        best_lr, t_per_step = run_lr_search(
            model=model,
            train_dataloader=search_loader,
            initial_lr=training_args.learning_rate,
            hours_to_complete=train_request["hours_to_complete"],
            grad_accum_steps=training_args.gradient_accumulation_steps,
            max_grad_norm=training_args.max_grad_norm,
            dataloader_factory=_halve_batch_dataloader,
            steps_per_epoch=_steps_per_epoch,
        )

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
            # Sync best_lr AND t_per_step from rank 0: each rank measures its own
            # warmup, and downstream decisions (gauss subsample, epoch planning)
            # must be identical across ranks or the dataset/step counts diverge
            # and DDP deadlocks. 0.0 is the None sentinel.
            _sync = torch.tensor([best_lr, t_per_step or 0.0],
                                 device=next(model.parameters()).device)
            torch.distributed.broadcast(_sync, src=0)
            best_lr = _sync[0].item()
            t_per_step = _sync[1].item() if _sync[1].item() > 0 else None
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

    # ── KL step-time overhead ──
    # The LR search measures t_per_step WITHOUT the base model (it isn't loaded
    # yet), but KL adds a base forward + completion-token KL each step. Inflate
    # the measured t_per_step so coverage/epoch planning don't over-plan; the
    # wall-clock stop is still the hard guarantee, this just keeps the cosine
    # schedule cooling properly. ~1.4x ≈ one extra forward (no backward).
    if use_kl and t_per_step is not None:
        _kl_overhead = 1.4
        t_per_step = t_per_step * _kl_overhead
        log_info(f"[sn56][kl] t_per_step inflado x{_kl_overhead} p/ overhead do termo KL: {t_per_step:.3f}s")

    # ── Gaussian difficulty subsampling for low-coverage runs ──
    # Uses the REAL measured per-step time (t_per_step, now synced across ranks)
    # instead of the params heuristic; falls back to the heuristic only when the
    # search was skipped (cached LR / DeepSpeed) and no measurement exists. Runs
    # before epoch planning so it sees the reduced dataset.
    #
    # Selection: keep the MEDIUM-DIFFICULTY core (Gaussian around the median
    # completion length) — both extremes (trivial / noise-length) are the least
    # informative. Sizing: coverage already dictates how MANY samples the budget
    # sees; the subsample only changes WHICH ones. Repeats buy ~nothing over
    # fresh data (Muennighoff 2023), so size the kept set to what the budget
    # covers ONCE (target = coverage * n) — breadth over repetition.
    #
    # Operates on the RAW dataset and re-packs: PackedDataset precomputes its
    # blocks, so mutating after packing is a no-op (this gate was dead on the
    # packed path before). Deterministic across ranks: t_per_step is broadcast,
    # gaussian_subsample is seeded, so every rank builds the same dataset.
    # Recompute from the CURRENT batch — it (and t_per_step) may have been
    # halved on OOM during the search; using the original batch would
    # overestimate coverage ~2x per halving and under-subsample.
    _cur_eff_bs = max(1, training_args.per_device_train_batch_size
                      * training_args.gradient_accumulation_steps
                      * training_args.world_size)
    if t_per_step is not None:
        _cur_spe = max(1, len(train_ds) // _cur_eff_bs)
        _coverage = _hours * 3600 * 0.85 / (_cur_spe * t_per_step)
        _cov_src = "medido"
    else:
        _coverage = _est_coverage
        _cov_src = "heurística"
    _n_raw = len(_raw_train_ds.eval_dataset)
    # MEASURED coverage only: the heuristic fallback (cached-LR retries /
    # DeepSpeed) over-estimates step time, which under-estimates coverage and
    # would over-cut see-once data on exactly the runs we know least about.
    if t_per_step is not None and _coverage < 0.5 and _n_raw > 2000:
        from utility import gaussian_subsample
        # Coverage is measured in (packed) steps but is a unitless fraction of
        # one epoch, so it translates to raw samples directly.
        #
        # INVARIANT — breadth pad: coverage is deliberately computed on the FULL
        # hours budget even though the LR search already spent up to 20% of it,
        # so the target runs ~20-25% generous. Training must outpace the
        # measured t_per_step by that margin before any sample is seen twice;
        # the expected case sees ~80% of the core exactly once. Do NOT "fix"
        # this to remaining-time — that would remove the hedge and turn
        # estimation error into discarded fresh data.
        _target = max(2000, int(_coverage * _n_raw))
        if _target < _n_raw:
            _raw_train_ds.eval_dataset = gaussian_subsample(
                _raw_train_ds.eval_dataset, _target
            )
            if _train_is_packed:
                from monkeypatch import PackedDataset
                _t1 = datetime.datetime.now()
                train_ds = PackedDataset(
                    _raw_train_ds,
                    tokenizer,
                    max_input_length=max_length,
                    max_packed_size=training_args.max_packed_size,
                    min_item_num=min_data_size_num,
                    use_fa=use_fa_packing,
                )
                trainer.train_dataset = train_ds
                _t2 = datetime.datetime.now()
                log_info(f"[sn56][gauss] re-costura: {(_t2 - _t1).total_seconds():.1f}s")
            else:
                train_ds = _raw_train_ds
                trainer.train_dataset = train_ds
            log_info(
                f"[sn56][gauss] coverage={_coverage:.0%}<50% ({_cov_src}), "
                f"núcleo de {_target}/{_n_raw} amostras (~1 época no orçamento; "
                f"steps/epoch agora {len(train_ds) // _cur_eff_bs})"
            )

    # ── Time-aware epoch planning ──
    # Compute on rank 0, broadcast to all ranks to avoid NCCL deadlock.
    if t_per_step is not None and not _use_deepspeed:
        _epoch_info = torch.tensor([0.0, 0.0, 0.0], device=next(model.parameters()).device)

        if is_main_process(LOCAL_RANK):
            _now = datetime.datetime.now(datetime.timezone.utc)
            _end_dt = datetime.datetime.strptime(
                train_request["end_time"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            remaining_secs = (_end_dt - _now).total_seconds()
            achievable_steps = int(remaining_secs * 0.85 / t_per_step)
            # Recompute from current batch size — may have been halved during LR search OOM
            _current_eff_bs = max(1,
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size)
            steps_per_epoch = len(train_ds) // _current_eff_bs

            if steps_per_epoch > 0:
                # Cap epochs: 4 for small datasets (<10K samples), 3 otherwise.
                # Beyond this, models memorize rather than generalize.
                _max_epochs = 4.0 if len(train_ds) < 10000 else 3.0
                # +25%: t_per_step is now the REAL post-search per-step time, so
                # achievable is accurate (only the 0.85 buffer is conservative). A
                # 1.25 stretch just spends that buffer — it plans slightly past the
                # deadline so the run uses the full budget, while keeping cosine
                # progress ~94% at the timer cut (LR ~min_lr, properly cooled). A
                # bigger stretch (e.g. 1.5) would leave the cosine at ~0.33x peak.
                # save_before_remaining_time is the real stop; floor 1.3 epochs so
                # even a tight job commits to a meaningful run.
                optimal_epochs = round(max(1.3, min(_max_epochs, 1.25 * achievable_steps / steps_per_epoch)), 2)
                new_total_steps = int(steps_per_epoch * optimal_epochs)
                warmup = min(training_args.warmup_steps, max(10, int(0.03 * new_total_steps)))
                eval_steps = float(training_args.eval_steps)
                if is_final_run:
                    # Eval every 1/8 epoch — frequent overfit detection and a
                    # denser snapshot trajectory for averaging. Evals before
                    # 0.75 epochs are skipped in the callback; the 10%-of-budget
                    # eval-timing governor widens this again if evals are slow.
                    eval_steps = float(max(20, steps_per_epoch // 8))
                _epoch_info[0] = optimal_epochs
                _epoch_info[1] = float(warmup)
                _epoch_info[2] = eval_steps

                log_info(
                    f"[sn56][relogio] Epochs {training_args.num_train_epochs} -> {optimal_epochs} "
                    f"(t_per_step={t_per_step:.3f}s, remaining={remaining_secs:.0f}s, "
                    f"achievable={achievable_steps}, steps/epoch={steps_per_epoch}, "
                    f"warmup={warmup}, eval_steps={int(eval_steps)})"
                )

        if torch.distributed.is_initialized():
            torch.distributed.broadcast(_epoch_info, src=0)

        if _epoch_info[0] > 0:
            training_args.num_train_epochs = _epoch_info[0].item()
            training_args.warmup_steps = int(_epoch_info[1].item())
            if is_final_run:
                training_args.eval_steps = int(_epoch_info[2].item())
                training_args.save_steps = int(_epoch_info[2].item())

    # ── Dev-pass time budgeting ──
    # The end-time save now stops training, so the dev-pass actually runs — but
    # the stock T-3min trigger leaves it no room to finish. Move the trigger
    # earlier by the dev-pass's estimated cost: one epoch over dev at eval bs 1,
    # ~FLOP-equivalent per (packed) dev block to one train micro-batch row, so
    # per-block cost ≈ t_per_step / (grad_accum * per_device_bs). 2x safety for
    # the lost batch parallelism at bs 1, +120s for the weights-only save.
    # Deterministic across ranks (t_per_step was broadcast), so the trigger
    # fires in lockstep. Trades minutes of cosine-tail training (~min_lr, worth
    # little) for a full pass over held-out data.
    _dev_pass_secs = None
    if is_final_run and _avg_mode == "ram" and len(dev_ds) > 0 and t_per_step is not None:
        _micro_cost = t_per_step / max(
            1,
            training_args.gradient_accumulation_steps
            * training_args.per_device_train_batch_size,
        )
        _blocks_per_rank = len(dev_ds) / max(1, training_args.world_size)
        _dev_pass_secs = 2.0 * _blocks_per_rank * _micro_cost + 120
        _extra_min = min(15.0, max(2.0, _dev_pass_secs / 60))
        when_to_eval_handler.save_before_remaining_time = (
            train_request["save_before_remaining_time"] + _extra_min
        )
        log_info(
            f"[sn56][devfit] orçamento ~{_dev_pass_secs:.0f}s "
            f"({len(dev_ds)} blocos dev @ micro={_micro_cost:.3f}s/rank); "
            f"parada final em T-{when_to_eval_handler.save_before_remaining_time:.1f}min"
        )

    # ── Wire the KL reference model (after LR search / epoch planning) ──
    # LoRA: no separate model — compute_loss reads base logits via
    # model.disable_adapter() (exact + zero extra memory). Full-FT: load a frozen
    # copy of the original model on this rank's device, like the evaluator does.
    if use_kl and not training_args.use_lora:
        _kl_attn = "flash_attention_2" if not training_args.disable_fa else "eager"
        if training_args.use_attn_implementation:
            _kl_attn = training_args.use_attn_implementation
        _kl_device = next(model.parameters()).device
        trainer.base_model = load_base_model(train_request["model_path"], _kl_attn, _kl_device)
        log_info(f"[sn56][kl] modelo base congelado carregado em {_kl_device} (full-FT)")
    elif use_kl:
        log_info("[sn56][kl] usando disable_adapter do LoRA como base (sem cópia)")

    log_info(f"fiscalização a cada {training_args.eval_steps} passos, {len(dev_ds)} na prova")
    trainer.train()

    # ── Final dev-data pass (data maximization on small datasets) ──
    # The dev split was held out only for selection; the scored test set is
    # separate. So reclaim it: one low-LR epoch over dev from the best checkpoint,
    # then weights-only save. Best-effort — any failure leaves the best
    # checkpoint untouched. Gated to RAM averaging (model reliably holds best
    # weights after train) and to datasets where dev is a meaningful fraction.
    try:
        _dev_n = len(dev_ds)
        # Window check: only start a dev-pass that can finish. The atomic swap
        # in dev_pass protects the submission either way; this just avoids
        # burning the final minutes on a doomed pass (e.g. when training
        # stopped via overfit early-stop near the deadline, or the budgeting
        # above never ran on a cached-LR/DeepSpeed path).
        _dev_window_ok = True
        if _dev_pass_secs is not None:
            _end_dt = datetime.datetime.strptime(
                train_request["end_time"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            _remaining = (_end_dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            _dev_window_ok = _remaining > _dev_pass_secs
            if not _dev_window_ok:
                log_info(
                    f"[sn56][devfit] sem janela ({_remaining:.0f}s restantes < "
                    f"~{_dev_pass_secs:.0f}s necessários), pulando"
                )
        if is_final_run and _avg_mode == "ram" and _dev_n > 0 and _dev_window_ok:
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
            log_info(
                f"[sn56][devfit] {_dev_n} amostras de dev, 1 época @ lr={_min_lr:.2e}"
            )
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
            log_info(
                f"[sn56][devfit] pulando (final={is_final_run}, modo={_avg_mode}, "
                f"dev={_dev_n})"
            )
    except Exception as _e:
        log_info(f"[sn56][devfit] dev-pass falhou ({_e}), mantendo melhor checkpoint")

    if is_main_process(LOCAL_RANK):
        success_file = os.path.join(training_args.output_dir, "success.txt")
        with open(success_file, "w") as f:
            f.write("Success")
    log_info("missão cumprida", "finish")

if __name__ == "__main__":
    main()
