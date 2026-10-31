"""The Triton NF4 dequantization is bit-identical to bitsandbytes, and so is a Linear4bit with it."""

import pytest
import torch

from solution import nf4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("shape", [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960),
                                   (3, 320)])
@pytest.mark.parametrize("nested", [True, False])
def test_dequant_is_bit_identical(shape, nested):
    import bitsandbytes.functional as bnbf

    torch.manual_seed(0)
    w = torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.05
    q, qs = bnbf.quantize_4bit(w, blocksize=64, compress_statistics=nested, quant_type="nf4")
    ref = bnbf.dequantize_4bit(q, qs)
    ours = nf4.dequantize_nf4(q, qs, torch.float16)
    assert ours.shape == ref.shape and ours.dtype == ref.dtype
    assert torch.equal(ours, ref)


def test_linear4bit_forward_backward_identical():
    import bitsandbytes as bnb

    nf4.patch_bnb_matmul_4bit()
    torch.manual_seed(0)
    k, n = 1536, 640
    lin = bnb.nn.Linear4bit(k, n, bias=True, compute_dtype=torch.float16,
                            compress_statistics=True, quant_type="nf4")
    lin.weight = bnb.nn.Params4bit(torch.randn(n, k, dtype=torch.float16) * 0.05,
                                   requires_grad=False, quant_type="nf4",
                                   compress_statistics=True, blocksize=64)
    lin.bias = torch.nn.Parameter(torch.randn(n), requires_grad=False)  # fp32, as in training
    lin = lin.cuda()

    # 4 x 512 rows (> 1536): base bitsandbytes takes its dequantize + F.linear path too.
    x = torch.randn(4, 512, k, device="cuda")  # fp32 input, as the RMSNorm output is
    dy = torch.randn(4, 512, n, device="cuda")

    def run(enabled):
        nf4.ENABLED = enabled
        xi = x.clone().requires_grad_(True)
        y = lin(xi)
        (y * dy).sum().backward()
        return y.detach(), xi.grad

    try:
        y_ref, g_ref = run(False)
        y_new, g_new = run(True)
    finally:
        nf4.ENABLED = True
    assert y_new.dtype == y_ref.dtype
    assert torch.equal(y_new, y_ref)
    assert torch.equal(g_new, g_ref)


def test_launch_cache_follows_the_weight():
    """The cached launch arguments are reused for the same weight (also through a fresh .t()
    view, as Linear4bit passes it) and rebuilt when a quant_state's weight changes."""
    import bitsandbytes.functional as bnbf

    torch.manual_seed(0)
    w = torch.randn(256, 1536, device="cuda", dtype=torch.float16) * 0.05
    q, qs = bnbf.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type="nf4")
    ref = bnbf.dequantize_4bit(q, qs)
    assert torch.equal(nf4.dequantize_nf4(q, qs), ref)
    first = nf4._LAUNCH[id(qs)]
    assert torch.equal(nf4.dequantize_nf4(q.t(), qs), ref)
    assert nf4._LAUNCH[id(qs)] is first
    assert nf4._weight_ok_cached(q.t(), qs) and nf4._weight_ok_cached(q.t(), qs)

    q2 = q.clone()
    q2[:7] ^= 0x5A  # another weight under the same quant_state object
    ref2 = bnbf.dequantize_4bit(q2, qs)
    assert torch.equal(nf4.dequantize_nf4(q2, qs), ref2)
    assert nf4._LAUNCH[id(qs)] is not first
