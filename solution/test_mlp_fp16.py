"""The fp16 MLP gives the same values and input gradients as the fp32-input MLP, bit for bit."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _linear4bit(k, n):
    import bitsandbytes as bnb

    lin = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.float16,
                            compress_statistics=True, quant_type="nf4")
    lin.weight = bnb.nn.Params4bit(torch.randn(n, k, dtype=torch.float16) * 0.05,
                                   requires_grad=False, quant_type="nf4",
                                   compress_statistics=True, blocksize=64)
    return lin


@pytest.mark.parametrize("tokens", [300, 2048])  # below and above bnb's gemm_4bit M threshold
def test_mlp_fp16_is_bit_identical(tokens):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution import mlp_fp16
    from solution.swiglu import swiglu

    mlp_fp16.patch_qwen2_mlp_fp16()
    torch.manual_seed(0)
    h, i = 256, 896
    mlp = Qwen2MLP(Qwen2Config(hidden_size=h, intermediate_size=i, hidden_act="silu"))
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = _linear4bit(h, i), _linear4bit(h, i), _linear4bit(i, h)
    mlp = mlp.cuda()

    x = torch.randn(1, tokens, h, device="cuda")  # fp32, as the RMSNorm output
    dy = torch.randn(1, tokens, h, device="cuda")

    # The current solution path: fp32 input into each Linear4bit, example SwiGLU.
    xr = x.clone().requires_grad_(True)
    y_ref = mlp.down_proj(swiglu(mlp.gate_proj(xr), mlp.up_proj(xr)))
    (y_ref * dy).sum().backward()

    xn = x.clone().requires_grad_(True)
    y_new = mlp(xn)  # patched forward
    (y_new.float() * dy).sum().backward()

    assert y_ref.dtype == torch.float32 and y_new.dtype == torch.float16
    assert torch.equal(y_new.float(), y_ref)
    assert xn.grad.dtype == torch.float32
    assert torch.equal(xn.grad, xr.grad)
