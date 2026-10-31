"""Convergence: a few dozen optimizer steps with our patches follow the original code's loss curve.

The unit tests compare one forward/backward. Here a tiny Qwen2 with LoRA on q/v (dropout on) is
trained twice from the same weights on the same right-padded batches with the same seed: once
through the original code (unpadding off, fused head not armed, reference RMSNorm) and once with
our RMSNorm, fused head + loss, MLP unpadding, packed layers and varlen attention. The per-step
losses must stay together as check 6 of the harness asks (stricter here), and both must learn.
"""

import copy

import pytest
import torch

from solution.test_packed import _model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

STEPS = 40


def _batches(b=4, t=48, vocab=1000, distinct=8):
    g = torch.Generator().manual_seed(0)
    data = []
    for _ in range(distinct):
        ids = torch.randint(0, vocab, (b, t), generator=g)
        lengths = torch.randint(t // 3, t + 1, (b,), generator=g)
        lengths[0] = t  # one full row, as the longest row of a batch is
        mask = (torch.arange(t)[None, :] < lengths[:, None]).long()
        labels = ids.masked_fill(mask == 0, -100)
        labels[:, : t // 5] = -100  # a prompt: no loss on it, as in assistant-only SFT
        data.append((ids.cuda(), mask.cuda(), labels.cuda()))
    return [data[i % distinct] for i in range(STEPS)]


def _reference_rmsnorm(self, hidden_states):  # Qwen2RMSNorm.forward, transformers 5.16.1
    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    return self.weight * h.to(input_dtype)


def _train(model, patched, batches):
    from solution import head, unpad

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)
    torch.manual_seed(1234)  # LoRA dropout draws the same masks in both runs
    losses = []
    for ids, mask, labels in batches:
        unpad.ENABLED, head._CTX.armed = patched, patched
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
        finally:
            unpad.ENABLED, head._CTX.armed = True, False
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(out.loss.item())
    return losses


def test_training_follows_the_original_loss_curve(monkeypatch):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from solution import head, packed, rmsnorm, unpad

    rmsnorm.patch_qwen2_rmsnorm()
    head.patch_qwen2_fused_head()
    unpad.patch_qwen2_unpad()
    packed.patch_qwen2_packed_layers()
    packed._VARLEN_STATE.update(ok=True, calls=0)

    ref_model = _model()
    new_model = copy.deepcopy(ref_model)
    batches = _batches()

    with monkeypatch.context() as m:  # the original RMSNorm for the reference run only
        m.setattr(Qwen2RMSNorm, "forward", _reference_rmsnorm)
        ref = _train(ref_model, False, batches)
    new = _train(new_model, True, batches)

    assert packed._VARLEN_STATE["ok"] and packed._VARLEN_STATE["calls"] > 0
    rel = [abs(a - b) / abs(a) for a, b in zip(ref, new)]
    assert max(rel) < 1e-2, f"per-step loss differs by up to {max(rel):.2%}"
    assert sum(rel) / len(rel) < 3e-3, f"mean per-step difference {sum(rel) / len(rel):.2%}"
    first, last = sum(ref[:8]) / 8, sum(ref[-8:]) / 8
    assert last < first, "the reference did not learn: the test would prove nothing"
    assert sum(new[-8:]) / 8 < sum(new[:8]) / 8
