"""Compile our Triton kernels while Soup loads and quantizes the model, not in training step 1.

Every measured run starts with empty compiler caches, and base Soup's first step took 0.4 s more
than its median while ours took 4.3 s more: that is the compilation of our kernels, one after
another, on the first launches. Loading the model takes much longer than that and does not need
these kernels, so a thread started at the end of apply() compiles them meanwhile
(JITFunction.warmup: compile only, no launch), and the trainer waits for it before its first step.

Nothing is cached across runs and nothing runs outside this process: the compilation is the same
work, timed inside the same `soup train` command, only earlier. Each warm-up uses the dtypes,
constexprs, num_warps and integer divisibility of the real launches, so the first real launch
finds the kernel compiled; a guess that does not match only costs the compilation it would have
cost anyway, and any failure here is ignored (the kernels then compile on first use, as before).
"""

from __future__ import annotations

import threading

import torch

ENABLED = True
_THREAD: threading.Thread | None = None

# Qwen2.5-1.5B-Instruct: hidden 1536, rms_norm_eps 1e-6 (the RMSNorm kernel takes both as
# constexprs). The other integers are runtime arguments; multiples of 16 match the real ones.
HIDDEN, RMS_EPS = 1536, 1e-06


def _compile_all(strict=False):
    import triton

    from . import head, nf4, rmsnorm, swiglu

    dev = torch.device("cuda", torch.cuda.current_device())

    def t(dtype, n=64):
        return torch.empty(n, dtype=dtype, device=dev)

    f16, f32 = torch.float16, torch.float32
    jobs = [
        lambda: rmsnorm._rmsnorm_forward.warmup(
            t(f32), t(f32), t(f32), t(f32), HIDDEN, RMS_EPS, triton.next_power_of_2(HIDDEN),
            grid=(1,), num_warps=4),
        lambda: rmsnorm._rmsnorm_backward_input.warmup(
            t(f32), t(f32), t(f32), t(f32), t(f32), HIDDEN, triton.next_power_of_2(HIDDEN),
            grid=(1,), num_warps=4),
        lambda: nf4.nf4_dequant_kernel.warmup(
            t(torch.uint8), t(f32), t(f32), t(f16), 16 * 1024,
            HALF_BLOCK=32, BLOCK=nf4.BLOCK, grid=(1,), num_warps=4),
        lambda: swiglu.example_swiglu_fwd_kernel.warmup(
            t(f16), t(f16), t(f16), 16 * 1024, BLOCK=swiglu.BLOCK, grid=(1,)),
        lambda: swiglu.example_swiglu_bwd_kernel.warmup(
            t(f16), t(f16), t(f16), t(f16), t(f16), 16 * 1024, BLOCK=swiglu.BLOCK, grid=(1,)),
    ]
    for write_grad in (True, False):
        jobs.append(lambda wg=write_grad: head.fused_head_ce_kernel.warmup(
            t(f16), 151936, t(torch.int64), t(f32), t(f32), t(torch.int32), 151936,
            IGNORE=head.IGNORE_INDEX, WRITE_GRAD=wg, GRAD_SCALE=head.GRAD_SCALE,
            BLOCK=head.BLOCK, grid=(1,), num_warps=8))
    for job in jobs:
        try:
            job()
        except Exception:  # noqa: BLE001 - that kernel simply compiles on first use
            if strict:
                raise


def start():
    """Start compiling in the background (called at the end of apply())."""
    global _THREAD
    if not ENABLED or _THREAD is not None or not torch.cuda.is_available():
        return
    _THREAD = threading.Thread(target=_compile_all, name="solution-triton-warmup", daemon=True)
    _THREAD.start()


def wait():
    """Block until the warm-up is done (before the first training step)."""
    if _THREAD is not None:
        _THREAD.join()
