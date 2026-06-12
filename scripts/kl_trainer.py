"""KL-regularised instruct trainer.

Mirrors the G.O.D evaluator's scoring for KL-regularised instruct tasks
(branch feature/instruct-kl-training). When a tournament instruct task is
flagged KL, the miner container receives ``USE_KL=1`` and ``KL_COEF=<float>``
and the validator scores:

    weighted_loss = eval_loss + kl_coef * KL(finetuned || base)

where the KL term is (validator/evaluation/eval_instruct_text.py
::_calculate_instruct_kl_divergence):

    kl_per_token = sum_v p_ft * (log p_ft - log p_base)      # over the vocab
    KL           = mean(kl_per_token  over positions where label != -100)

We reproduce the validator's definition *exactly* rather than "fix" it, because
it is the quantity we are ranked on:

  * NOT next-token shifted. The KL at position i uses logits[i] (the distribution
    over token i+1) but masks on label[i]. The CE term it is added to *is*
    shifted, so the KL is off-by-one relative to its own CE. As a drift penalty
    it is still well-defined; matching it is what aligns training with scoring.
  * Computed in the model's native (bf16) dtype, no float upcast.

The base-model reference logits come from one of two sources, chosen for memory
safety:

  * LoRA  -> ``model.disable_adapter()`` on the policy model itself. The LoRA
    base is byte-identical to the original weights, so this is exact and costs
    ZERO extra GPU memory (no second model).
  * full-FT -> a frozen full copy of the original model (``load_base_model``),
    the same source the evaluator loads. Unavoidable when there is no adapter to
    disable.
"""
import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer

from utility import log_info


def load_base_model(
    model_path: str,
    attn_implementation: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> transformers.PreTrainedModel:
    """Load a frozen copy of the original (base) model for the KL reference.

    Mirrors how the evaluator loads ``original_model`` as the base: a plain
    causal-LM in bf16, no LoRA, eval mode, gradients off. Placed on ``device``
    (the training rank's device) so KL forwards stay on-GPU. Only used for the
    full fine-tune path; LoRA uses ``disable_adapter`` instead.
    """
    base = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    base.to(device)
    base.eval()
    try:
        base.config.use_cache = False
    except Exception:
        pass
    for p in base.parameters():
        p.requires_grad_(False)
    return base


class KLRegularizedTrainer(Trainer):
    """Trainer that adds ``kl_coef * KL(finetuned || base)`` on completion tokens.

    Drop-in for the stock ``Trainer`` used by train_instruct: only ``compute_loss``
    is overridden, so every callback, the LR search, checkpoint averaging, the
    dev pass, NEFTune, PLW/dedup and the wall-clock stop behave identically. When
    KL is inactive (no coef, or neither base source configured) it falls back to
    plain cross-entropy, so it is safe to use unconditionally.

    Args:
        kl_coef: coefficient applied to the KL term (the task's ``KL_COEF``).
        base_model: frozen reference model for the full-FT path; ``None`` for LoRA.
        use_lora_base: when True, take base logits from ``self.model.disable_adapter()``
            instead of ``base_model``.
    """

    def __init__(self, *args, kl_coef: float = 0.0, base_model=None, use_lora_base: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_coef = kl_coef
        self.base_model = base_model
        self.use_lora_base = use_lora_base
        self._kl_logged = False
        # Gradient-accumulation scale correctness. In transformers>=4.46,
        # training_step skips the `loss / grad_accum_steps` division when
        # model_accepts_loss_kwargs is True (it then relies on the model's loss
        # having divided by the window-global num_items_in_batch). We forward
        # WITHOUT labels and return a per-micro-batch *mean* (CE+KL), so we must
        # take the division path instead — otherwise accumulated gradients are
        # ~grad_accum× too large and training runs far above the searched LR.
        # Forcing the flag False makes training_step divide by grad_accum, so our
        # per-micro-batch means accumulate to the correct windowed mean.
        self.model_accepts_loss_kwargs = False

    def _kl_active(self) -> bool:
        return bool(self.kl_coef) and (self.use_lora_base or self.base_model is not None)

    def _base_logits(self, input_ids, attention_mask):
        """Reference logits from the base model, always under no_grad."""
        with torch.no_grad():
            if self.use_lora_base:
                # Disable the LoRA adapters on the underlying PEFT model; the
                # forward then returns the original (base) model's logits. Use the
                # unwrapped self.model to sidestep DDP machinery on this no-grad pass.
                with self.model.disable_adapter():
                    return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            # Lazily co-locate the frozen base with the live batch. The base copy
            # is loaded before trainer.train(), where `model` may still be on CPU
            # (cached-LR / DeepSpeed paths skip the LR search that moves it to GPU);
            # Trainer only moves the policy to its device inside train(). By the
            # first compute_loss the inputs are on the training device, so anchor
            # the base to input_ids.device. One-shot: .to() is a no-op once placed.
            if self.base_model.device != input_ids.device:
                self.base_model.to(input_ids.device)
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        # Forward WITHOUT labels so logits are materialised even under liger
        # (liger's fused cross-entropy skips logits when labels are passed, and
        # we need full-vocab logits for the KL).
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Standard causal-LM cross-entropy (shifted), honouring label smoothing.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=getattr(self.args, "label_smoothing_factor", 0.0) or 0.0,
        )

        # MoE load-balancing aux loss: present in outputs when output_router_logits
        # is set (as load_lora_model does). Re-add it here since we computed CE
        # manually — matches the stock path's loss for MoE LoRA models.
        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None:
            ce_loss = ce_loss + getattr(self.model.config, "router_aux_loss_coef", 0.0) * aux_loss

        if self._kl_active():
            mask = labels != -100  # completion tokens (unshifted — mirrors validator)
            if mask.any():
                # KL only at completion positions: identical to the validator's
                # masked mean, but avoids materialising [B, T, V] probability
                # tensors over the whole sequence.
                sel_ft = logits[mask]  # [N, V], keeps grad
                base_logits = self._base_logits(input_ids, attention_mask)
                sel_base = base_logits[mask]  # [N, V]

                ft_log_probs = F.log_softmax(sel_ft, dim=-1)
                base_log_probs = F.log_softmax(sel_base, dim=-1)
                kl_per_token = (ft_log_probs.exp() * (ft_log_probs - base_log_probs)).sum(dim=-1)
                kl = kl_per_token.mean()
            else:
                kl = logits.new_zeros(())

            loss = ce_loss + self.kl_coef * kl
            if not self._kl_logged:
                log_info(
                    f"[sn56][kl] ativo kl_coef={self.kl_coef}, "
                    f"fonte={'lora' if self.use_lora_base else 'copia'}, "
                    f"ce={ce_loss.item():.4f}, kl={float(kl):.4f}, loss={loss.item():.4f}"
                )
                self._kl_logged = True
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss
