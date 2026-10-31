"""The fused head with the weight streamed from host RAM gives what the GPU-resident head gives."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)


def test_streamed_head_matches_resident():
    from solution import head

    torch.manual_seed(0)
    b, t, dim, v = 3, 50, 64, 20000  # three vocabulary pieces, the last one partial
    w = (torch.randn(v, dim, device="cuda") * 0.05).half()
    hidden = torch.randn(b, t, dim, device="cuda")
    labels = torch.randint(0, v, (b, t), device="cuda")
    labels[:, :10] = -100  # prompt
    mask = torch.ones(b, t, dtype=torch.long, device="cuda")
    mask[1, 40:] = 0
    labels[1, 40:] = -100

    def run(weight):
        h = hidden.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss, step = head.fused_head_loss(h, weight, labels, attention_mask=mask)
        loss.backward()
        return loss.detach(), step, h.grad

    loss_ref, step_ref, g_ref = run(w)
    host = torch.empty(w.shape, dtype=w.dtype, pin_memory=True)
    host.copy_(w)
    loss, step, g = run(host)
    torch.testing.assert_close(
        loss, loss_ref, rtol=1e-3, atol=1e-4
    )  # cuBLAS may round a logit differently
    torch.testing.assert_close(
        step.token_entropy, step_ref.token_entropy, rtol=1e-3, atol=2e-3
    )
    torch.testing.assert_close(
        step.token_loss, step_ref.token_loss, rtol=1e-3, atol=2e-3
    )
    assert abs(int(step.correct) - int(step_ref.correct)) <= 1
    torch.testing.assert_close(
        g, g_ref, rtol=2e-2, atol=2e-3 * g_ref.abs().max().item()
    )


def test_streamed_head_flat_logits_do_not_overflow():
    """Nearly equal logits over a whole piece with same-sign weights: an unnormalised
    sum of 256 * exp(z - m) over a piece times W would overflow fp16; the gradient must stay
    finite and match the resident head."""
    from solution import head

    torch.manual_seed(0)
    b, t, dim, v = 2, 24, 64, 20000
    w = (0.5 + 0.01 * torch.randn(v, dim, device="cuda")).half()
    hidden = 0.01 * torch.randn(b, t, dim, device="cuda")
    labels = torch.randint(0, v, (b, t), device="cuda")
    mask = torch.ones(b, t, dtype=torch.long, device="cuda")

    def run(weight):
        h = hidden.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = head.fused_head_loss(h, weight, labels, attention_mask=mask)
        loss.backward()
        return loss.detach(), h.grad

    loss_ref, g_ref = run(w)
    host = torch.empty(w.shape, dtype=w.dtype, pin_memory=True)
    host.copy_(w)
    loss, g = run(host)
    assert torch.isfinite(g).all()
    torch.testing.assert_close(loss, loss_ref, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(g, g_ref, rtol=2e-2, atol=2e-3 * g_ref.abs().max().item())
