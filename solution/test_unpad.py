"""Unpadding must give the same loss and the same gradients as the padded model."""

import pytest
import torch

from solution import unpad

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_unpadded_call_matches_masked_reference():
    torch.manual_seed(0)
    b, t, h = 3, 7, 16
    lin = torch.nn.Linear(h, 24, device=DEVICE)
    x = torch.randn(b, t, h, device=DEVICE, requires_grad=True)
    mask = torch.ones(b, t, dtype=torch.long, device=DEVICE)
    mask[0, 4:] = 0
    mask[2, 2:] = 0
    idx = unpad.token_index(mask, b, t)
    assert idx is not None

    y = unpad.unpadded_call(lin, x, idx)
    ref = lin(x) * mask[..., None]
    torch.testing.assert_close(y, ref)
    g = torch.randn_like(y)
    (gx,) = torch.autograd.grad((y * g).sum(), x)
    (rx,) = torch.autograd.grad((ref * g).sum(), x)
    torch.testing.assert_close(gx, rx)
    assert (gx[mask == 0] == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tiny_qwen2_loss_and_grads_match(dtype):
    if dtype == torch.float16 and DEVICE == "cpu":
        pytest.skip("fp16 matmuls need CUDA")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    unpad.patch_qwen2_unpad()
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = Qwen2ForCausalLM(cfg).to(DEVICE, dtype).train()

    ids = torch.randint(0, 128, (4, 20), device=DEVICE)
    mask = torch.ones_like(ids)
    for row, length in enumerate([20, 13, 6, 17]):  # right padding, as TRL does
        mask[row, length:] = 0
    labels = ids.masked_fill(mask == 0, -100)

    def run(enabled):
        unpad.ENABLED = enabled
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        logits = out.logits.detach()[mask.bool()]
        return out.loss.detach(), logits, grads

    try:
        loss_ref, logits_ref, grads_ref = run(False)
        loss, logits, grads = run(True)
    finally:
        unpad.ENABLED = True

    tol = dict(rtol=1e-4, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(loss, loss_ref, **tol)
    torch.testing.assert_close(logits, logits_ref, **tol)
    assert grads.keys() == grads_ref.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], grads_ref[name], **tol, msg=name)
