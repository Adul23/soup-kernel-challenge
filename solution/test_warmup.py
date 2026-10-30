"""The warm-up compiles every kernel with the same specialization as the real launches."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _cache_size(kernel):
    caches = getattr(kernel, "device_caches", None)
    if not caches:
        return None
    entry = caches.get(torch.cuda.current_device())
    return None if entry is None else len(entry[0])


def test_warmup_runs_and_real_launches_reuse_it():
    from solution import head, nf4, warmup

    warmup._compile_all(strict=True)  # every job compiles without error

    before = (_cache_size(nf4.nf4_dequant_kernel), _cache_size(head.fused_head_ce_kernel))
    import bitsandbytes.functional as bnbf

    w = torch.randn(256, 1536, device="cuda", dtype=torch.float16)
    q, qs = bnbf.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type="nf4")
    nf4.dequantize_nf4(q, qs, torch.float16)
    hidden = torch.randn(2, 9, 64, device="cuda")
    weight = torch.randn(151936, 64, device="cuda") * 0.05
    labels = torch.randint(0, 151936, (2, 9), device="cuda")
    head.fused_head_loss(hidden.requires_grad_(True), weight, labels)[0].backward()
    after = (_cache_size(nf4.nf4_dequant_kernel), _cache_size(head.fused_head_ce_kernel))
    if None in before or None in after:
        pytest.skip("this Triton keeps its kernel cache elsewhere")
    assert after == before, "a real launch compiled a new variant: warm-up arguments differ"
