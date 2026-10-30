"""Packed decoder layers match the padded model, with LoRA dropout drawing the same numbers."""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _model():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=160,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128)
    model = Qwen2ForCausalLM(cfg).cuda().train()
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():  # LoRA_B starts at zero; make the adapters (and dropout) matter
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.05)
    return model.train()


def test_packed_layers_match_padded_and_keep_rng():
    from solution import packed, unpad

    unpad.patch_qwen2_unpad()
    packed.patch_qwen2_packed_layers()
    ref_model = _model()
    new_model = copy.deepcopy(ref_model)

    ids = torch.randint(0, 1000, (4, 40), device="cuda")
    mask = torch.ones_like(ids)
    for row, n in enumerate([40, 23, 9, 31]):
        mask[row, n:] = 0
    labels = ids.masked_fill(mask == 0, -100)

    seen = []
    layer = new_model.base_model.model.model.layers[1]
    hook = layer.register_forward_pre_hook(lambda mod, args: seen.append(args[0].dim()))

    def run(model, enabled):
        unpad.ENABLED = enabled
        try:
            torch.manual_seed(1234)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
        finally:
            unpad.ENABLED = True
        rng = torch.cuda.get_rng_state()
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        return out, grads, rng

    try:
        ref, grads_ref, rng_ref = run(ref_model, False)
        seen.clear()
        new, grads_new, rng_new = run(new_model, True)
    finally:
        hook.remove()

    assert seen == [2], "layer 1 did not get the packed [n_real, hidden] stream"
    assert torch.equal(rng_new, rng_ref), "LoRA dropout drew a different amount of randomness"
    real = mask.bool()
    torch.testing.assert_close(new.logits[real].float(), ref.logits[real].float(),
                               rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(new.loss, ref.loss, rtol=1e-3, atol=1e-4)
    assert grads_new.keys() == grads_ref.keys() and any("lora_A" in n for n in grads_ref)
    scale = max(g.abs().max().item() for g in grads_ref.values())
    for name, g in grads_ref.items():
        torch.testing.assert_close(grads_new[name], g, rtol=2e-2,
                                   atol=2e-3 * max(g.abs().max().item(), 1e-2 * scale) + 1e-8,
                                   msg=name)
