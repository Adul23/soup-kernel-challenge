"""Triton RMSNorm against a plain-PyTorch reference, forward and backward.

The reference is written out here instead of calling Qwen2RMSNorm.forward: the patch is
class-level, so once it is applied every Qwen2RMSNorm module, the "reference" one included, runs
the Triton kernel and would only be compared with itself. Picked up by `python -m pytest solution`.
"""

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from solution import rmsnorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference(module, x):
    """Qwen2RMSNorm.forward of transformers 5.16.1, written out (independent of any patch)."""
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + module.variance_epsilon)
    return module.weight * h.to(input_dtype)


# fp32 input is what training sees: the tied embedding is kept in fp32, so the residual stream
# (the norm's input) is fp32 in every layer. fp16 covers the rounding path of the kernel.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(32, 1536), (4, 128, 1536), (3, 37, 1536)])
@pytest.mark.parametrize("weight_grad", [False, True])
def test_rmsnorm_matches_reference(shape, dtype, weight_grad):
    rmsnorm.patch_qwen2_rmsnorm()
    assert Qwen2RMSNorm._soup_custom_rmsnorm

    torch.manual_seed(7)
    n = shape[-1]
    norm = Qwen2RMSNorm(n, eps=1e-6).cuda().float()
    with torch.no_grad():
        norm.weight.uniform_(0.8, 1.2)
    norm.weight.requires_grad_(weight_grad)

    x_ref = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    dy = torch.randn(*shape, device="cuda", dtype=torch.float32)

    y_ref = reference(norm, x_ref)
    (y_ref * dy).sum().backward()
    dw_ref = norm.weight.grad.clone() if weight_grad else None
    norm.weight.grad = None

    y_new = norm(x_new)  # patched class forward: the Triton kernels
    (y_new * dy).sum().backward()

    assert y_new.dtype == y_ref.dtype and x_new.grad.dtype == x_ref.grad.dtype
    tol = dict(atol=1e-5, rtol=1e-5) if dtype == torch.float32 else dict(atol=4e-3, rtol=4e-3)
    torch.testing.assert_close(y_new, y_ref, **tol)
    gtol = dict(atol=1e-4, rtol=1e-4) if dtype == torch.float32 else dict(atol=1.5e-2, rtol=1.5e-2)
    torch.testing.assert_close(x_new.grad, x_ref.grad, **gtol)
    if weight_grad:
        torch.testing.assert_close(norm.weight.grad, dw_ref, atol=2e-2, rtol=2e-3)
