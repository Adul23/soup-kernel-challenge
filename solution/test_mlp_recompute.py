"""The recomputing MLP is bit-identical to the mlp_fp16 path and saves only its fp16 input."""

import pytest
import torch

from solution.test_mlp_fp16 import _linear4bit

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _saved_bytes(fn):
    sizes = []

    def pack(t):
        sizes.append(t.numel() * t.element_size())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, sum(sizes)


@pytest.mark.parametrize("tokens", [300, 2048])
def test_recompute_mlp_bit_identical_and_lighter(tokens):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution import mlp_fp16, mlp_recompute, nf4
    from solution.swiglu import swiglu

    nf4.patch_bnb_matmul_4bit()
    mlp_fp16.patch_qwen2_mlp_fp16()
    mlp_recompute.patch_qwen2_mlp_recompute()
    torch.manual_seed(0)
    h, i = 256, 896
    mlp = Qwen2MLP(Qwen2Config(hidden_size=h, intermediate_size=i, hidden_act="silu"))
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = _linear4bit(h, i), _linear4bit(h, i), _linear4bit(i, h)
    mlp = mlp.cuda()

    x = torch.randn(1, tokens, h, device="cuda")
    dy = torch.randn(1, tokens, h, device="cuda")

    # Reference: the mlp_fp16 path written out (two casts, Linear4bit calls, example SwiGLU).
    xr = x.clone().requires_grad_(True)

    def ref():
        xa, xb = xr.to(torch.float16), xr.to(torch.float16)
        return mlp.down_proj(swiglu(mlp.gate_proj(xa), mlp.up_proj(xb)))

    y_ref, saved_ref = _saved_bytes(ref)
    (y_ref.float() * dy).sum().backward()

    xn = x.clone().requires_grad_(True)
    y_new, saved_new = _saved_bytes(lambda: mlp(xn))  # patched forward: RecomputeMLP
    (y_new.float() * dy).sum().backward()

    assert y_new.dtype == y_ref.dtype == torch.float16
    assert torch.equal(y_new, y_ref)
    assert xn.grad.dtype == torch.float32
    assert torch.equal(xn.grad, xr.grad)
    assert saved_new <= tokens * h * 2          # x16 only
    assert saved_ref >= 2 * tokens * i * 2      # gate and up
