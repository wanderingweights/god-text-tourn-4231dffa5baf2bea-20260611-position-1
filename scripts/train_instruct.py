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
import quasar_loader

import warnings

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
    if quasar_loader.is_quasar(model_path):
        # QuasarLong needs sdpa (hybrid branches) + raven/fla on path + meta fill.
        # q_lora/liger are not supported for this custom arch.
        model = quasar_loader.load_quasar_model(model_path)
    else:
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
    if quasar_loader.is_quasar(model_path):
        # Quasar's modeling calls transformers' deprecated AttentionMaskConverter
        # API, firing a FutureWarning on EVERY forward that floods the logs. Silence
        # just that line, and ONLY on the Quasar path so other models' warnings are
        # untouched. Harmless model-author deprecation, not ours to fix.
        warnings.filterwarnings(
            "ignore", message=r"The attention mask API under", category=FutureWarning
        )
        _m = quasar_loader.load_quasar_model(model_path)
        # Freeze the handful of structurally-unused params (disabled local-window
        # branch + quasar-branch SSM params the fla scan never backprops) so ZeRO-3
        # full fine-tune doesn't try to reduce a None grad and crash. The CE loss
        # already covers every param that affects the forward (grad-probe verified).
        quasar_loader.freeze_unused_params(_m)
        # transformers 5.x save_pretrained expects _tied_weights_keys as a DICT; this
        # model declares it as a LIST (old format) -> remove_tied_weights_from_state_dict
        # crashes ("'list' object has no attribute 'keys'") on EVERY checkpoint save.
        # tie_word_embeddings=False here (untied), so clear the list-form attr on any
        # module that has it -> save skips tied-weight removal and writes all weights.
        for _mod in _m.modules():
            if isinstance(getattr(_mod, "_tied_weights_keys", None), list):
                _mod._tied_weights_keys = None
        return _m

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

    is_quasar = quasar_loader.is_quasar(train_request["model_path"])
    if is_quasar:
        # Custom tokenizer (AutoTokenizer can't parse its TokenizersBackend
        # config) and make fla/raven importable before any model load.
        quasar_loader.prepare(train_request["model_path"])
        tokenizer = quasar_loader.load_tokenizer(train_request["model_path"])
        # Sample-packing is now CORRECT for Quasar: the model build isolates packed
        # segments (block-diagonal SDPA mask + per-segment cu_seqlens, BOTH derived from
        # position_ids that reset per segment). So enable the NAIVE packer
        # (monkeypatch.pack_data_points_naive): concat segments, position_ids = arange
        # per segment, labels[0] = -100 per segment (kills the cross-segment boundary
        # target), trailing pad with position_ids=0. NOT the FA packer (no flash-attn on
        # Blackwell). Buffer = max_length (forced to 4096 below). grad-accum stays at the
        # instruct_config default (doesn't affect tok/s).
        # CAVEATS: (a) needs the isolating model build deployed (box's model dir is
        # pristine); (b) a sample > buffer overflows pack_data_points_naive's fixed-len
        # assert -> the >buffer "giants" need the own-buffer/bs=1 path (follow-up; none
        # in the <=2021 test split).
        training_args.packing = True
        train_request["packing_mode"] = "naive"
        # Bypassing the LR search (below) removes its OOM-driven 8->4 batch halving, so
        # pin the batch that fits the 4096 buffer (bs=8 = 32768 tok/microbatch OOMs).
        training_args.per_device_train_batch_size = 2
        # NOTE: do NOT set ddp_find_unused_parameters=True. Quasar's vectorized MoE
        # computes ALL experts every step, so there are zero unused params (DDP confirms:
        # "did not find any unused parameters in the forward pass"). The flag only buys an
        # extra autograd-graph traversal per iteration (slower). Leave it at the default
        # (off), same as every other model on this script.
        # Target total batch = per_device(2) x grad_accum(16) x world(2 GPUs) = 64 packed
        # seqs/optimizer-step. At 8k ctx, bs=2 keeps per-microbatch tokens (2x8192) equal
        # to the old bs=4x4096, so per-step memory holds ~constant; accum doubled 8->16 to
        # hold the effective batch at 64 sequences. Adjust if the GPU count changes.
        training_args.gradient_accumulation_steps = 16
    else:
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
    if is_quasar:
        # Pack-buffer length (PackedDataset.max_input_length). 8k recovers the long
        # reasoning samples that 4k dropped (~16% of the set). MUST equal the tokenize
        # drop threshold (instruct_config sets train_request["max_length"]=8192 for
        # Quasar). Samples > buffer still overflow pack_data_points_naive's fixed-length
        # assert (the >8192 "giants" need the own-buffer/bs=1 path — follow-up).
        max_length = 8192

    # we already tokenize the data and save it to .pt (torch format, fast)
    # Quasar: MyDataset yields NATURAL length (pad=False); the custom collator (below)
    # pads each batch to a FIXED max_length (one stable shape -> triton kernels cache)
    # and counts real-vs-pad tokens for the throughput log. Non-Quasar keeps in-dataset
    # fixed-length padding.
    _pad_items = not is_quasar
    train_ds = MyDataset(
        tokenizer,
        f"datasets/train_tokenized_{task_id}.json",
        max_length,
        pad=_pad_items,
    )

    dev_ds = MyDataset(
        tokenizer,
        f"datasets/dev_tokenized_{task_id}.json",
        max_length,
        pad=_pad_items,
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
    _sharded = _shard_ds or bool(getattr(training_args, "fsdp", None))
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
    # Quasar collator: pad each batch to a FIXED max_length (ONE stable shape so triton
    # /fla kernels compile once and cache; variable shapes => recompile thrash). Tallies
    # real vs padded tokens into _tput for the throughput callback below.
    import time as _time
    _tput = {"real": 0, "pad": 0, "t": None}
    _pad_log = {"n": 0}
    def _quasar_pad_collator(features):
        # Packed rows (PackedDataset) are exactly max_length and carry position_ids (the
        # per-segment reset = the isolation signal) + labels (segment starts already
        # -100). Pad to a FIXED max_length = ONE stable shape (triton cache); PRESERVE
        # position_ids (key-aware); tally real-vs-pad tokens from attention_mask for
        # [sn56][tput]. (Pre-packing/unpacked rows simply lack position_ids -> skipped.)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        right = tokenizer.padding_side != "left"
        pad_to = max_length
        fill = {"input_ids": pad_id, "attention_mask": 0, "labels": -100, "position_ids": 0}
        keys = [k for k in fill if k in features[0]]
        def _pad(t, val):
            d = pad_to - int(t.shape[0])
            if d < 0:
                return t[:pad_to]
            if d == 0:
                return t
            tail = t.new_full((d,), val)
            return torch.cat([t, tail] if right else [tail, t])
        batch = {k: torch.stack([_pad(f[k], fill[k]) for f in features]) for k in keys}
        real = int(batch["attention_mask"].sum()); total = int(batch["attention_mask"].numel())
        _tput["real"] += real
        _tput["pad"] += (total - real)
        _pad_log["n"] += 1
        if _pad_log["n"] <= 6 or _pad_log["n"] % 100 == 0:
            log_info("[sn56][pad] batch#%d bs=%d pad_to=%d real=%d/%d (%.0f%% real, packed=%s)"
                     % (_pad_log["n"], len(features), pad_to, real, total,
                        100.0 * real / max(1, total), "position_ids" in features[0]))
        return batch

    from transformers import TrainerCallback as _TrainerCB
    class _ThroughputCB(_TrainerCB):
        # Logs real tok/s every optimizer step (grad_accum=1 -> every microbatch) so
        # throughput is visible from step 1 instead of hand-counting log lines.
        def on_step_end(self, args, state, control, **kw):
            now = _time.time()
            if _tput["t"] is not None:
                dt = now - _tput["t"]
                tot = _tput["real"] + _tput["pad"]
                if dt > 0 and tot > 0:
                    log_info("[sn56][tput] step %d: %.2fs/step | %.0f real tok/s | %.0f tok/s w/pad | %.0f%% pad"
                             % (state.global_step, dt, _tput["real"] / dt, tot / dt, 100.0 * _tput["pad"] / tot))
            _tput["t"] = now
            _tput["real"] = 0
            _tput["pad"] = 0

    class _HFCkptUpload(_TrainerCB):
        # Crash-safety: push each saved checkpoint to HF (latest-only / overwrite) so a
        # box failure doesn't lose the trained model — our direct text_trainer runs skip
        # job_handler's HF push, so without this saves are local-only. Repo from env
        # HF_CKPT_REPO, token from env HUGGINGFACE_TOKEN/HF_TOKEN (no secret in-script);
        # no-op if unset. Uploads model files only (skips optimizer.pt); never raises.
        def __init__(self):
            self.repo = os.environ.get("HF_CKPT_REPO", "")
            tok = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
            self.api = None
            if self.repo and tok:
                try:
                    from huggingface_hub import HfApi
                    self.api = HfApi(token=tok)
                    self.api.create_repo(self.repo, repo_type="model", private=False, exist_ok=True)
                    log_info(f"[sn56][hf] checkpoint uploads -> {self.repo} (latest-only)")
                except Exception as e:
                    log_info(f"[sn56][hf] upload disabled (repo init failed: {e})")
                    self.api = None
            else:
                log_info("[sn56][hf] checkpoint upload OFF (set HF_CKPT_REPO + HUGGINGFACE_TOKEN to enable)")

        def on_save(self, args, state, control, **kw):
            if self.api is None:
                return
            import glob as _glob
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt):
                cks = [c for c in _glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
                       if c.rsplit("-", 1)[-1].isdigit()]
                ckpt = max(cks, key=lambda c: int(c.rsplit("-", 1)[-1])) if cks else None
            if not ckpt or not os.path.isdir(ckpt):
                log_info(f"[sn56][hf] no checkpoint dir to upload at step {state.global_step}")
                return
            try:
                _t0 = _time.time()
                self.api.upload_folder(
                    repo_id=self.repo, folder_path=ckpt, repo_type="model",
                    allow_patterns=["*.safetensors", "*.json", "*.jinja", "tokenizer*", "*.model", "*.txt"],
                    commit_message=f"checkpoint step {state.global_step}",
                )
                log_info(f"[sn56][hf] uploaded checkpoint-{state.global_step} -> {self.repo} ({_time.time()-_t0:.0f}s)")
            except Exception as e:
                log_info(f"[sn56][hf] upload FAILED (step {state.global_step}): {e}")

    class _EarlyFailSaveCB(_TrainerCB):
        # Early fail-check ONLY: force a SINGLE checkpoint at step 10 so a broken save
        # path surfaces in minutes instead of after hours. Leaves the normal save
        # schedule (CustomEvalSaveCallback / ckpt_avg) and training dynamics (grad-accum)
        # completely untouched. Set on ALL ranks so DDP agrees on the save step.
        def on_step_end(self, args, state, control, **kw):
            if state.global_step == 10:
                control.should_save = True
            return control

    class _MoEUtilCB(_TrainerCB):
        # MoE utilization -> wandb (rank-0, aggregated over all MoE layers). Non-invasive:
        # forward hooks read each block's topk_idx (routed expert choices) + output norms,
        # so no edits to silx's model code. Env MOE_LOG=0 disables. Per-expert load
        # histogram every `hist_every` log calls (heavier than the scalars).
        def __init__(self, num_experts, hist_every=10):
            self.num_experts = int(num_experts)
            self.hist_every = hist_every
            self.counts = None        # [num_experts] cumulative routed-slot counts
            self.shared_norm = 0.0    # sum ||shared_expert_out||
            self.total_norm = 0.0     # sum ||moe_block_out|| (routed+shared)
            self._handles = []
            self._log_calls = 0

        def _moe_hook(self, module, inp, out):
            # out = (y_total, (router_logits, topk_idx)); topk_idx [bsz, seq, top_k]
            try:
                idx = out[1][1].reshape(-1)
                c = torch.bincount(idx, minlength=self.num_experts).double()
                if self.counts is None:
                    self.counts = torch.zeros(self.num_experts, dtype=torch.float64, device=c.device)
                self.counts += c
                self.total_norm += float(out[0].detach().float().norm())
            except Exception:
                pass

        def _shared_hook(self, module, inp, out):
            try:
                self.shared_norm += float(out.detach().float().norm())
            except Exception:
                pass

        def on_train_begin(self, args, state, control, model=None, **kw):
            if model is None:
                return
            for m in model.modules():
                if hasattr(m, "moe_vectorized"):
                    self._handles.append(m.register_forward_hook(self._moe_hook))
                    if hasattr(m, "shared_experts"):
                        self._handles.append(m.shared_experts.register_forward_hook(self._shared_hook))
            log_info(f"[sn56][moe] utilization logging on ({len(self._handles)} hooks)")

        def on_log(self, args, state, control, logs=None, **kw):
            try:
                import wandb
            except Exception:
                return
            if self.counts is None or wandb.run is None:
                return
            counts = self.counts
            total = counts.sum().clamp_min(1.0)
            p = (counts / total).clamp_min(1e-12)
            ent = float(-(p * p.log()).sum() / torch.log(torch.tensor(float(self.num_experts))))
            metrics = {
                "moe/dead_expert_frac": float((counts == 0).sum()) / self.num_experts,
                "moe/load_cv": float(counts.std() / (counts.mean() + 1e-12)),
                "moe/routing_entropy": ent,                       # 1.0 = uniform routing
                "moe/max_expert_load": float(counts.max() / total),
                "moe/shared_expert_share": self.shared_norm / (self.total_norm + 1e-12),
            }
            self._log_calls += 1
            if self._log_calls % max(1, self.hist_every) == 0:
                try:
                    metrics["moe/expert_load_hist"] = wandb.Histogram(
                        sequence=(counts / total).cpu().tolist())
                except Exception:
                    pass
            try:
                wandb.log(metrics, step=state.global_step)
            except Exception:
                pass
            self.counts = None
            self.shared_norm = 0.0
            self.total_norm = 0.0

        def on_train_end(self, args, state, control, **kw):
            for h in self._handles:
                try:
                    h.remove()
                except Exception:
                    pass

    _trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,  # transformers 5.x renamed Trainer(tokenizer=) -> processing_class
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=(_quasar_pad_collator if is_quasar else None),
        callbacks=[
            # Early fail-check: ONE forced save at step 10 (only when SAVE_SMOKE=1).
            # All ranks (control flag must agree across DDP).
            *([_EarlyFailSaveCB()] if os.environ.get("SAVE_SMOKE") == "1" else []),
            # MoE utilization -> wandb (rank-0, env MOE_LOG=0 to disable).
            *([_MoEUtilCB(getattr(getattr(model, "config", None), "num_experts", 256))]
              if (is_quasar and is_main_process(LOCAL_RANK)
                  and os.environ.get("MOE_LOG", "1") != "0") else []),
            # Rank-0 only: under DDP both ranks would otherwise race the HF upload
            # (duplicate/conflicting commits) and double-log throughput. Trainer writes
            # the checkpoint on the main process, so rank 0 is exactly where on_save sees it.
            *([_ThroughputCB()] if (is_quasar and is_main_process(LOCAL_RANK)) else []),
            *([] if ckpt_avg is None else [ckpt_avg]),
            # HF upload AFTER ckpt_avg: ckpt_avg.on_save rewrites checkpoint-N in place
            # with the souped/averaged best weights (when an avg is the current best), so
            # uploading after it pushes the SOUP — not the raw step weights — to HF.
            # Kept before CustomEvalSaveCallback so the checkpoint dir is still present.
            *([_HFCkptUpload()] if is_main_process(LOCAL_RANK) else []),
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
    elif is_quasar and os.environ.get("LR_FINDER") != "1":
        # Bypass the (slow, hours-long) empirical LR search for Quasar from BASE — it
        # consistently lands ~1.6e-4 (S1 best: lr=1.60e-04, loss=1.16). For a WARM START
        # (continuing from a saved checkpoint) set LR_FINDER=1: this branch is skipped and
        # we fall through to the live search (the right LR differs from base — usually
        # lower). The search uses the 8-bit optimizer (OOM fix) in the else branch below.
        training_args.learning_rate = 1.6e-4
        log_info(f"[sn56][farejando] BYPASS p/ Quasar — lr fixo={training_args.learning_rate:.2e} (melhor S1 da busca)")
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
        # Run the search with the SAME optimizer as training. lr_search defaults to
        # fp32 AdamW, whose m+v states are 4x the 8-bit paged optimizer training uses
        # (~144GB vs ~36GB for 18B trainable) — that 4x is exactly what OOMs the search
        # at bs=4 and forces the throughput-killing batch halving. Matching it also
        # calibrates the found LR on the real optimizer. (None -> lr_search's default.)
        _search_opt_cls, _search_opt_kwargs = None, None
        if "8bit" in (getattr(training_args, "optim", "") or "").lower():
            try:
                import bitsandbytes as _bnb
                _search_opt_cls = (_bnb.optim.PagedAdamW8bit if "paged" in training_args.optim.lower()
                                   else _bnb.optim.AdamW8bit)
                _search_opt_kwargs = {"weight_decay": training_args.weight_decay}
                log_info(f"[sn56][farejando] busca usa optim de treino: {_search_opt_cls.__name__} (evita OOM do AdamW fp32)")
            except Exception as _e:
                log_info(f"[sn56][farejando] optim 8-bit p/ busca indisponivel ({_e}); AdamW fp32 padrao")
        best_lr, t_per_step = run_lr_search(
            model=model,
            train_dataloader=search_loader,
            initial_lr=training_args.learning_rate,
            hours_to_complete=train_request["hours_to_complete"],
            grad_accum_steps=training_args.gradient_accumulation_steps,
            max_grad_norm=training_args.max_grad_norm,
            dataloader_factory=_halve_batch_dataloader,
            steps_per_epoch=_steps_per_epoch,
            optimizer_cls=_search_opt_cls,
            optimizer_kwargs=_search_opt_kwargs,
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
