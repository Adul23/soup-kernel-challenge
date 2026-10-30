"""Fused head + loss against the dense path base Soup runs (fp16 logits under autocast)."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from solution import head

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _batch(b=3, t=37, v=9001, dim=64, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(v, dim, generator=g) / dim ** 0.5).cuda()
    hidden = torch.randn(b, t, dim, generator=g).cuda() * 3
    lengths = [t, t - 9, t - 20][:b]
    mask = torch.zeros(b, t, dtype=torch.long)
    labels = torch.randint(0, v, (b, t), generator=g)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
        labels[i, n:] = -100
        labels[i, : 5 + 3 * i] = -100  # prompt
    return hidden, w, labels.cuda(), mask.cuda()


def _dense(hidden, w, labels, mask, den, scale):
    """What base Soup computes: autocast fp16 logits, ForCausalLMLoss, TRL metrics."""
    h = hidden.detach().clone().requires_grad_(True)
    logits = h.half() @ w.half().t()          # fp16 output, fp32 accumulation
    z = logits.float()
    target = F.pad(labels, (0, 1), value=-100)[:, 1:]
    v = z.shape[-1]
    token_ce = F.cross_entropy(z.reshape(-1, v), target.reshape(-1), ignore_index=-100,
                               reduction="none").view_as(target)
    loss = token_ce.sum() / den
    (loss * scale).backward()
    logp = z.detach().log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    lab = target != -100
    correct = ((z.detach().argmax(-1) == target) & lab).sum()
    return loss.detach(), token_ce.detach(), entropy, correct, h.grad


@cuda
@pytest.mark.parametrize("scale", [1.0, 65536.0])  # 65536 = GradScaler's initial scale
@pytest.mark.parametrize("chunk", [16, 512])
def test_fused_head_matches_dense(scale, chunk):
    hidden, w, labels, mask = _batch()
    target = F.pad(labels, (0, 1), value=-100)[:, 1:]
    den = (target != -100).sum()
    loss_r, tok_r, ent_r, cor_r, grad_r = _dense(hidden, w, labels, mask, den, scale)

    h = hidden.detach().clone().requires_grad_(True)
    loss, step = head.fused_head_loss(h, w, labels, mask, num_items_in_batch=den, chunk=chunk)
    (loss * scale).backward()

    real = mask.bool()
    lab = target != -100
    torch.testing.assert_close(loss, loss_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(step.token_loss[lab], tok_r[lab], rtol=1e-4, atol=1e-4)
    assert (step.token_loss[~lab] == 0).all()
    torch.testing.assert_close(step.token_entropy[real], ent_r[real], rtol=1e-4, atol=1e-4)
    assert abs(int(step.correct) - int(cor_r)) <= 1  # fp16 near-ties between two GEMM shapes
    assert int(step.labelled) == int(lab.sum())
    tol = 2e-3 * grad_r.abs().max().item()
    torch.testing.assert_close(h.grad, grad_r, rtol=2e-2, atol=tol)
    assert (h.grad[~lab] == 0).all()  # no label at t: nothing flows back from the head


@cuda
@pytest.mark.parametrize("m,v,d", [(37, 9001, 64), (300, 151936, 1536), (1, 130, 1536)])
def test_triton_gemms_match_torch(m, v, d):
    torch.manual_seed(0)
    a = torch.randn(m, d, device="cuda").half()
    w = (torch.randn(v, d, device="cuda") / d ** 0.5).half()
    out = torch.empty(m, v, dtype=torch.float16, device="cuda")
    head.head_logits(out, a, w)
    ref = a @ w.t()
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    g = (torch.randn(m, v, device="cuda") / v ** 0.5).half()
    dh = head.head_grad(g, w)
    ref_dh = g.float() @ w.float()
    assert dh.dtype == torch.float32
    torch.testing.assert_close(dh, ref_dh, rtol=1e-3, atol=1e-3)
    assert head._GEMM["triton"], "Triton GEMMs fell back to cuBLAS: check 4 would lose them"


@cuda
def test_argmax_ties_take_first_index_across_blocks():
    v, dim = 9001, 64
    w = torch.zeros(v, dim, device="cuda")
    u = torch.randn(dim, device="cuda")
    w[5] = u
    w[9000] = u  # same logit as id 5, in the third vocabulary block
    hidden = (u / u.norm())[None, None, :].expand(2, 9, dim).contiguous() * 4
    labels = torch.full((2, 9), 5, device="cuda")
    labels[1] = 9000
    mask = torch.ones(2, 9, dtype=torch.long, device="cuda")
    _, step = head.fused_head_loss(hidden, w, labels, mask, chunk=4)
    assert int(step.correct) == 8  # row 0: 8 positions predict 5 and are labelled 5; row 1: none


@cuda
def test_tiny_qwen2_under_autocast_matches_unpatched():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    head.patch_qwen2_fused_head()
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=5003, hidden_size=64, intermediate_size=160,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg).cuda().float().train()
    model.lm_head.weight.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(False)

    ids = torch.randint(0, cfg.vocab_size, (4, 24), device="cuda")
    mask = torch.ones_like(ids)
    for row, n in enumerate([24, 15, 9, 20]):
        mask[row, n:] = 0
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, :4] = -100
    den = (labels[:, 1:] != -100).sum()

    def run(armed):
        model.zero_grad(set_to_none=True)
        head._CTX.armed, head._CTX.step = armed, None
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=mask, labels=labels,
                            num_items_in_batch=den, use_cache=False)
        finally:
            head._CTX.armed = False
        out.loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        return out, grads, head._CTX.step

    ref, grads_ref, step_ref = run(False)
    new, grads_new, step = run(True)
    assert step_ref is None and step is not None and new.logits is None

    torch.testing.assert_close(new.loss, ref.loss, rtol=2e-3, atol=1e-4)
    assert grads_new.keys() == grads_ref.keys()
    for name, g in grads_ref.items():
        torch.testing.assert_close(grads_new[name], g, rtol=2e-2,
                                   atol=2e-3 * g.abs().max().item() + 1e-8, msg=name)

    # Soup's rewind rows: per-row mean CE from the dense logits vs from step.token_loss.
    z = ref.logits.float()
    rows_ref = [F.cross_entropy(z[i, :-1], labels[i, 1:], ignore_index=-100).item()
                for i in range(z.shape[0])]
    tokens = (labels[:, 1:] != -100).sum(1)
    rows_new = (step.token_loss.sum(1) / tokens.clamp(min=1)).tolist()
    assert rows_new == pytest.approx(rows_ref, rel=1e-3)


def test_metrics_follow_trl_formulas():
    torch.manual_seed(0)
    b, t, v = 2, 7, 50
    logits = torch.randn(b, t, v)
    labels = torch.randint(0, v, (b, t))
    labels[:, :2] = -100
    mask = torch.ones(b, t, dtype=torch.long)
    mask[1, 5:] = 0
    labels[1, 5:] = -100

    trainer = SimpleNamespace(accelerator=SimpleNamespace(gather_for_metrics=lambda x: x),
                              _metrics={"train": defaultdict(list)}, _total_train_tokens=10)
    head.log_logit_metrics(trainer, "train", {"attention_mask": mask}, labels, logits)
    m = trainer._metrics["train"]
    assert list(m) == ["entropy", "num_tokens", "mean_token_accuracy"]

    logp = logits.log_softmax(-1)  # TRL 0.29.1, written out
    ent = (-(logp.exp() * logp).sum(-1) * mask).sum() / mask.sum()
    shift = labels[:, 1:]
    acc = ((logits[:, :-1].argmax(-1) == shift) & (shift != -100)).sum() / (shift != -100).sum()
    assert m["entropy"][0] == pytest.approx(ent.item(), rel=1e-6)
    assert m["num_tokens"] == [10 + int(mask.sum())]
    assert m["mean_token_accuracy"][0] == pytest.approx(acc.item(), rel=1e-6)
