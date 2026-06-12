from model_utility import get_model_architecture, get_model_num_params, get_use_liger, disable_flash_attention, get_gradient_checkpointing, get_gpu_count
from copy import deepcopy
from lr_finder import estimate_starting_lr
from adaptive_max_length import compute_max_length

DPO_CONFIG = {
    "0_1_b": {
        "lr": 1.35e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 16,
    },
    "1_2_b": {
        "lr": 8.7e-6,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 12,
    },
    "2_4_b": {
        "lr": 6.5e-6,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 12,
        "use_lora": True
    },
    "4_5_b": {
        "lr": 6.25e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 12,
        "use_lora": True
    },
    "5_9_b": {
        "lr": 7.5e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 8,
        "use_lora": True
    },
    "9_12_b": {
        "lr": 5e-6,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 32,
        "gradient_checkpointing": False
    },
    "12_14_b": {
        "lr": 8.5e-6,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 24,
        "gradient_checkpointing": False
    },
    "14_15_b": {
        "lr": 8.5e-6,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 18,
        "gradient_checkpointing": False
    },
    "15_40_b": {
        "lr": 8e-6,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 16,
        "gradient_checkpointing": False
    },
    "40_80_b": {
        "lr": 8e-6,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 8,
        "gradient_checkpointing": False
    }        
}

for key in DPO_CONFIG:
    DPO_CONFIG[key]["label"] = key
    

def get_config(param_nums: int) -> dict:
    result = None
    if param_nums < 1_000_000_000:
        result = DPO_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        result = DPO_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        result = DPO_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        result = DPO_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        result = DPO_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        result = DPO_CONFIG["9_12_b"]
    elif param_nums < 14_000_000_000:
        result = DPO_CONFIG["12_14_b"]
    elif param_nums < 15_000_000_000:  
        result = DPO_CONFIG["14_15_b"]
    elif param_nums < 35_000_000_000:
        result = DPO_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        result = DPO_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported", flush=True)
        result = {
            "lr": 4e-5,
            "distributed": "ds",
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True
        }
    if param_nums < 4_000_000_000 and param_nums > 1_330_000_000:
        result["gpu_count"] = 2
    if param_nums > 13_330_000_000: # 8 GPUs for 13.3B
        result["gpu_count"] = 8
    return result


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")
    gpu_nums = get_gpu_count()
    start_cmd = "python"
    run_type = config.get("distributed", "ddp")
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_dpo.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size {batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_steps {warmup_steps} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} --disable_fa {disable_fa} \
    --max_length {max_length} --max_prompt_length {max_prompt_length} \
    --dataloader_pin_memory True"""
    )

    if config.get("use_lora", False):
        template += (
            " --use_peft --lora_r 128 --lora_alpha 256 --lora_target_modules all-linear"
        )

    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))
    
    if config.get("use_attn_implementation", ""):
        use_attn_implementation = config["use_attn_implementation"]
        template = template + f""" --use_attn_implementation {use_attn_implementation}"""
        
    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    config = get_config(param_nums)

    # Adaptive max_length from dataset sequence length distribution.
    # Can go above defaults when model supports it.
    model_max_pos = None
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        model_max_pos = getattr(cfg, "max_position_embeddings", None)
    except Exception:
        pass

    baseline_stats = train_info.get("baseline_stats")
    if baseline_stats is not None:
        dataset_stats = baseline_stats.get("dataset", {})
        seq_dist = dataset_stats.get("seq_length_distribution")
        max_length = compute_max_length(
            seq_dist, default=1024, packing=False,
            model_max_length=model_max_pos,
        )
        chosen_dist = dataset_stats.get("chosen_length_distribution")
        if chosen_dist and seq_dist:
            chosen_p99 = chosen_dist.get("p99", 0)
            total_p99 = seq_dist.get("p99", 0)
            estimated_prompt_p99 = max(total_p99 - chosen_p99, total_p99 // 2)
            prompt_dist = {"p99": estimated_prompt_p99, "p50": estimated_prompt_p99 // 2}
            max_prompt_length = compute_max_length(prompt_dist, default=512, packing=False)
        else:
            max_prompt_length = min(512, max_length // 2)
    else:
        max_length = 1024
        max_prompt_length = 512

    # Time-aware warmup: ~60 steps/hour cap, refined in train script with 3% ratio
    hours = train_info.get("hours_to_complete", 2)
    warmup_steps = max(10, min(200, int(hours * 60)))

    # Scale batch size down when max_length is high to avoid OOM.
    # DPO processes chosen+rejected (2x sequences), so memory scales fast.
    # Reference: bs=12 fits at max_length=1024 on 80GB GPU for 1.5B model.
    _ref_maxlen = 1024
    _bs = config["batch_size"]
    if max_length > _ref_maxlen:
        _scale = _ref_maxlen / max_length
        _bs = max(1, int(_bs * _scale))
        print(f"[sn56][dpo-bs] max_length={max_length} > {_ref_maxlen}, batch {config['batch_size']} -> {_bs}", flush=True)

    run_config = {
        "epoch_num": 3,
        "batch_size": _bs,
        "learning_rate": config["lr"],
        "min_lr_rate": 0.25,
        "warmup_steps": warmup_steps,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": 1,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "use_attn_implementation": "kernels-community/vllm-flash-attn3" if train_info.get("is_openai", False) else ""
    }
    
    if not config.get("gradient_checkpointing", True):
        run_config["gradient_checkpointing"] = False
    
    total_batch_size = run_config["batch_size"] * run_config["gpu_nums"]
    if total_batch_size < 64:
        run_config["gradient_accumulation_steps"] = min(4, int(64 / total_batch_size))
    
    if train_info["find_lk_lr"]:
        effective_bs = run_config["batch_size"] * run_config["gradient_accumulation_steps"] * run_config["gpu_nums"]
        lr = estimate_starting_lr(
            train_info.get("baseline_stats"),
            "DpoTask",
            param_nums,
            effective_bs,
            fallback_lr=run_config["learning_rate"],
        )
        if lr is not None:
            # DPO's risk is asymmetric: too-low LR is recoverable (the in-train
            # held-out LR search nudges it back up), too-high LR collapses the
            # policy irrecoverably. So the finder may go BELOW the hand-tuned
            # bucket LR (useful per-model adaptivity) but never ABOVE it — the
            # bucket value is the hottest LR we trust for this size.
            run_config["learning_rate"] = min(lr, config["lr"])

    run_config["learning_rate"] *= train_info["reg_ratio"]
    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])
    if run_config["disable_fa"] == "False":
        run_cmd = run_cmd + " --padding_free True"
    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 3
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500
    train_request["checking_step"] = 80
    
    return {
        "train_request": train_request,
        "run_cmd": run_cmd
    }
