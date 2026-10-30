"""Original fused Qwen2 RMSNorm implementation for the Soup Kernel Challenge.

Written for the hackathon: Triton forward and input-gradient kernels.
The FP16 -> FP32 -> FP16 -> FP32 rounding order is intentionally preserved.
Test with the official Kaggle T4 harness before using in a submission.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_forward(X, W, Y, INV_RMS, N: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < N

    x = tl.load(X + row * N + col, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + col, mask=mask, other=0).to(tl.float32)
    avg_sq = tl.sum(x * x, 0) / N
    inv_rms = tl.rsqrt(avg_sq + EPS)
    normalized = (x * inv_rms).to(X.dtype.element_ty).to(tl.float32)
    y = normalized * w

    tl.store(Y + row * N + col, y, mask=mask)
    tl.store(INV_RMS + row, inv_rms)


@triton.jit
def _rmsnorm_backward_input(X, W, DY, INV_RMS, DX, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < N

    x = tl.load(X + row * N + col, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + col, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + row * N + col, mask=mask, other=0).to(tl.float32)
    inv_rms = tl.load(INV_RMS + row)

    # The reference multiplies a half-precision normalized tensor by FP32 weights.
    # The gradient of that half-precision operand is rounded to input dtype.
    grad_norm = (dy * w).to(X.dtype.element_ty).to(tl.float32)
    grad_norm = tl.where(mask, grad_norm, 0.0)
    dot = tl.sum(grad_norm * x, 0) / N
    dx = (grad_norm - x * (inv_rms * inv_rms) * dot) * inv_rms
    tl.store(DX + row * N + col, dx, mask=mask)


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        n = x.shape[-1]
        rows = x.numel() // n
        out_dtype = torch.promote_types(x.dtype, weight.dtype)
        out = torch.empty(x.shape, device=x.device, dtype=out_dtype)
        inv_rms = torch.empty((rows,), device=x.device, dtype=torch.float32)
        b = triton.next_power_of_2(n)
        _rmsnorm_forward[(rows,)](x, weight, out, inv_rms, n, eps, b, num_warps=4)
        ctx.save_for_backward(x, weight, inv_rms)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, inv_rms = ctx.saved_tensors
        n = x.shape[-1]
        rows = x.numel() // n
        grad_x = None
        grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_x = torch.empty_like(x)
            dy = grad_out.contiguous()
            _rmsnorm_backward_input[(rows,)](
                x, weight, dy, inv_rms, grad_x,
                n, triton.next_power_of_2(n), num_warps=4,
            )

        if ctx.needs_input_grad[1]:
            # RMSNorm weights are normally frozen during this LoRA task.
            # Keep the general gradient path correct for independent validation.
            norm = (x.reshape(rows, n).to(torch.float32) * inv_rms[:, None])
            norm = norm.to(x.dtype).to(torch.float32)
            grad_weight = (grad_out.reshape(rows, n).to(torch.float32) * norm).sum(dim=0)
            grad_weight = grad_weight.to(weight.dtype)

        return grad_x, grad_weight, None


def patch_qwen2_rmsnorm():
    """Patch only Qwen2RMSNorm.forward; no task or trainable-parameter changes."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls, "_soup_custom_rmsnorm", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        if (
            not hidden_states.is_cuda
            or not hidden_states.is_contiguous()
            or hidden_states.dtype not in (torch.float16, torch.float32)
            or self.weight.dtype != torch.float32
            or self.weight.ndim != 1
            or self.weight.numel() != hidden_states.shape[-1]
        ):
            return original(self, hidden_states)
        return _FusedRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)

    cls.forward = forward
    cls._soup_custom_rmsnorm = True