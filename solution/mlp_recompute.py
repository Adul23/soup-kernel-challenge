"""Qwen2 MLP that keeps only its fp16 input for backward and recomputes gate/up there.

After mlp_fp16 the MLP still keeps gate and up (fp16, [tokens, 8960] each) for the SwiGLU
backward: about 72 MB per layer at 2000 tokens, 2 GB over 28 layers at the peak. Here one autograd
function runs the whole MLP and saves only x16 ([tokens, 1536] fp16, 6x smaller). In backward
each NF4 weight is dequantized once by our Triton kernel (as bitsandbytes' backward does anyway)
and used twice for gate/up: to recompute the forward product and to take the input gradient.
The extra cost is two fp16 matmuls per layer; there is no extra dequantization.

Bit for bit the same as the mlp_fp16 path (test_mlp_recompute.py checks it with torch.equal):
the same F.linear on the same fp16 operands gives the same gate/up, the SwiGLU launches the same
kernels, and the input gradient is the two fp16 products widened to fp32 and summed in fp32, as
autograd does through the two casts of mlp_fp16.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
import triton

from . import nf4
from .mlp_fp16 import _is_fp16_linear4bit
from .swiglu import BLOCK as SWIGLU_BLOCK
from .swiglu import example_swiglu_bwd_kernel, example_swiglu_fwd_kernel

ENABLED = True


def _weight(layer, dtype):
    return nf4.dequantize_nf4(layer.weight, layer.weight.quant_state, dtype)


def _swiglu_fwd(gate, up):  # the same launch as solution.swiglu.SwiGLU.forward
    out = torch.empty_like(gate)
    n = gate.numel()
    example_swiglu_fwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](gate, up, out, n,
                                                              BLOCK=SWIGLU_BLOCK)
    return out


def _swiglu_bwd(grad, gate, up):  # the same launch as solution.swiglu.SwiGLU.backward
    dgate, dup = torch.empty_like(gate), torch.empty_like(up)
    n = gate.numel()
    example_swiglu_bwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](grad, gate, up, dgate, dup, n,
                                                              BLOCK=SWIGLU_BLOCK)
    return dgate, dup


class RecomputeMLP(torch.autograd.Function):
    """down(swiglu(gate(x16), up(x16))) with x16 = fp16(x); saves x16 only."""

    @staticmethod
    def forward(ctx, x, gate_proj, up_proj, down_proj):
        ctx.layers = (gate_proj, up_proj, down_proj)
        ctx.x_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x16 = x.to(torch.float16)
            gate = F.linear(x16, _weight(gate_proj, torch.float16)).contiguous()
            up = F.linear(x16, _weight(up_proj, torch.float16)).contiguous()
            h = _swiglu_fwd(gate, up)
            del gate, up
            out = F.linear(h, _weight(down_proj, torch.float16))
        ctx.save_for_backward(x16)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        (x16,) = ctx.saved_tensors
        gate_proj, up_proj, down_proj = ctx.layers
        with torch.autocast(device_type=x16.device.type, enabled=False):
            grad_out = grad_out.to(torch.float16)
            dh = torch.matmul(grad_out, _weight(down_proj, torch.float16)).contiguous()
            w_gate = _weight(gate_proj, torch.float16)
            gate = F.linear(x16, w_gate).contiguous()
            w_up = _weight(up_proj, torch.float16)
            up = F.linear(x16, w_up).contiguous()
            dgate, dup = _swiglu_bwd(dh, gate, up)
            del gate, up, dh
            grad_x = torch.matmul(dgate, w_gate).to(ctx.x_dtype)
            grad_x += torch.matmul(dup, w_up).to(ctx.x_dtype)
        return grad_x, None, None, None


def _usable(mlp, x) -> bool:
    if not (ENABLED and nf4.ENABLED and x.is_cuda and x.dtype == torch.float32 and x.numel()):
        return False
    if getattr(mlp.config, "hidden_act", "silu") != "silu":
        return False
    for layer in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        if not _is_fp16_linear4bit(layer) or layer.bias is not None:
            return False
        if not nf4.weight_ok(layer.weight, getattr(layer.weight, "quant_state", None)):
            return False
    return x.shape[-1] == mlp.gate_proj.weight.quant_state.shape[1]


def patch_qwen2_mlp_recompute():
    """Wraps the installed Qwen2MLP.forward; install after mlp_fp16, before unpad."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls, "_soup_mlp_recompute", False):
        return
    original = cls.forward

    @functools.wraps(original)
    def forward(self, x):
        if not _usable(self, x):
            return original(self, x)
        return RecomputeMLP.apply(x, self.gate_proj, self.up_proj, self.down_proj)

    cls.forward = forward
    cls._soup_mlp_recompute = True
