"""Tests for the KL-regularised instruct trainer.

Run: python scripts/test_helpers/test_kl_trainer.py
(plain script, no pytest dependency — mirrors the other test_helpers).

Covers:
  1. The KL term equals the validator's _calculate_instruct_kl_divergence
     byte-for-byte on identical logits (unshifted, label-masked, mean).
  2. compute_loss end-to-end on the full-FT path (frozen base copy):
     loss == ce + kl_coef * KL.
  3. The LoRA path takes base logits from self.model.disable_adapter()
     (verified with a stub adapter context manager; no peft needed).
  4. kl_coef == 0 / no base => pure cross-entropy fallback.
"""
import contextlib
import os
import sys
import tempfile
import types

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

# utility.py imports wandb at module load (present in the training container, not
# necessarily in a bare test env). Stub it so the import chain resolves.
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kl_trainer import KLRegularizedTrainer  # noqa: E402


def _tiny_model(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg).eval()


def _validator_kl(finetuned_logits, base_logits, labels) -> float:
    """Reproduces validator/evaluation/eval_instruct_text.py::_calculate_instruct_kl_divergence."""
    base_log_probs = F.log_softmax(base_logits, dim=-1)
    finetuned_log_probs = F.log_softmax(finetuned_logits, dim=-1)
    finetuned_probs = finetuned_log_probs.exp()
    kl_per_token = (finetuned_probs * (finetuned_log_probs - base_log_probs)).sum(dim=-1)
    mask = (labels != -100).float()
    total_kl = (kl_per_token * mask).sum().item()
    total_tokens = int(mask.sum().item())
    return total_kl / total_tokens if total_tokens else 0.0


def _manual_ce(logits, labels) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _make_trainer(model, **kw) -> KLRegularizedTrainer:
    with tempfile.TemporaryDirectory() as d:
        args = TrainingArguments(output_dir=d, report_to=[], use_cpu=True, per_device_train_batch_size=2)
        return KLRegularizedTrainer(model=model, args=args, **kw)


def _inputs():
    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :4] = -100  # prompt masked (completion-only KL/CE)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def test_full_ft_matches_validator():
    policy = _tiny_model(1)
    base = _tiny_model(2)  # different weights => nonzero KL
    for p in base.parameters():
        p.requires_grad_(False)
    kl_coef = 0.5
    trainer = _make_trainer(policy, kl_coef=kl_coef, base_model=base)

    inputs = _inputs()
    with torch.no_grad():
        loss = trainer.compute_loss(policy, inputs)
        logits = policy(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        base_logits = base(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

    ce = _manual_ce(logits, inputs["labels"]).item()
    kl = _validator_kl(logits, base_logits, inputs["labels"])
    expected = ce + kl_coef * kl
    assert kl > 0, f"expected nonzero KL, got {kl}"
    assert abs(loss.item() - expected) < 1e-4, f"loss {loss.item()} != ce+coef*kl {expected}"
    print(f"[ok] full-FT: loss={loss.item():.5f} == ce({ce:.5f})+{kl_coef}*kl({kl:.5f})")


def test_lora_path_uses_disable_adapter():
    """LoRA path: base logits must come from self.model.disable_adapter()."""
    policy = _tiny_model(1)
    base_state = _tiny_model(2)  # what 'adapter-off' should return

    called = {"n": 0}

    @contextlib.contextmanager
    def fake_disable_adapter():
        # Swap policy weights to the base state for the duration of the context.
        saved = {k: v.clone() for k, v in policy.state_dict().items()}
        policy.load_state_dict(base_state.state_dict())
        called["n"] += 1
        try:
            yield
        finally:
            policy.load_state_dict(saved)

    trainer = _make_trainer(policy, kl_coef=0.5, use_lora_base=True)
    policy.disable_adapter = fake_disable_adapter  # stub the PEFT API
    trainer.model = policy

    inputs = _inputs()
    with torch.no_grad():
        loss = trainer.compute_loss(policy, inputs)
        logits = policy(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        base_logits = base_state(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

    ce = _manual_ce(logits, inputs["labels"]).item()
    kl = _validator_kl(logits, base_logits, inputs["labels"])
    expected = ce + 0.5 * kl
    assert called["n"] == 1, "disable_adapter() was not used for base logits"
    assert abs(loss.item() - expected) < 1e-4, f"loss {loss.item()} != {expected}"
    print(f"[ok] lora: disable_adapter used; loss={loss.item():.5f} == ce+coef*kl({expected:.5f})")


def test_fallback_pure_ce():
    policy = _tiny_model(1)
    # kl_coef=0 -> no KL even with a base model set
    trainer = _make_trainer(policy, kl_coef=0.0, base_model=_tiny_model(2))
    inputs = _inputs()
    with torch.no_grad():
        loss = trainer.compute_loss(policy, inputs)
        logits = policy(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    ce = _manual_ce(logits, inputs["labels"]).item()
    assert abs(loss.item() - ce) < 1e-4, f"fallback loss {loss.item()} != ce {ce}"
    print(f"[ok] fallback: kl_coef=0 -> pure CE loss={loss.item():.5f}")


def test_grad_accum_scale_fix():
    """Regression for the grad-accumulation scale bug.

    In transformers>=4.46, training_step only divides the loss by grad_accum when
    model_accepts_loss_kwargs is False. We return a per-micro-batch mean and don't
    normalise by num_items_in_batch, so the trainer MUST force the flag False or
    accumulated gradients are ~grad_accum× too large (training above the searched
    LR). Pin the flag so a refactor / transformers bump can't silently revert it.
    """
    trainer = _make_trainer(_tiny_model(1), kl_coef=0.5, base_model=_tiny_model(2))
    assert trainer.model_accepts_loss_kwargs is False, (
        "model_accepts_loss_kwargs must be False so training_step divides by "
        "grad_accum (our compute_loss returns a per-micro-batch mean)"
    )
    # Also holds for the plain-CE fallback construction.
    plain = _make_trainer(_tiny_model(1), kl_coef=0.0)
    assert plain.model_accepts_loss_kwargs is False
    print("[ok] grad-accum: model_accepts_loss_kwargs forced False")


def test_grad_flows_through_kl():
    policy = _tiny_model(1).train()
    base = _tiny_model(2)
    for p in base.parameters():
        p.requires_grad_(False)
    trainer = _make_trainer(policy, kl_coef=0.5, base_model=base)
    loss = trainer.compute_loss(policy, _inputs())
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "no gradient flowed from KL+CE loss"
    print(f"[ok] grad: {len(grads)} param tensors received gradient")


if __name__ == "__main__":
    test_full_ft_matches_validator()
    test_lora_path_uses_disable_adapter()
    test_fallback_pure_ce()
    test_grad_accum_scale_fix()
    test_grad_flows_through_kl()
    print("\nALL KL TRAINER TESTS PASSED")
