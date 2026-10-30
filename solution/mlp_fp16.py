"""Qwen2 MLP in fp16 end to end: no fp32 copies of the 8960-wide activations, same numbers.

Base Soup: the MLP input is fp32 (the RMSNorm weight is fp32), so each bitsandbytes Linear4bit
casts it to fp16, multiplies in fp16 and casts the result back to fp32 (`.to(inp_dtype)`).
gate and up therefore live as fp32 [tokens, 8960] tensors, and the SwiGLU keeps both for
backward: about 4 GB at the peak over 28 layers, holding nothing but fp16 values.

Here the input is cast to fp16 once per projection (what bitsandbytes did inside), so gate, up,
the SwiGLU output and the down projection stay fp16. Every value is the same as in base Soup:
- gate/up: the same fp16 matmul of the same fp16 input, minus a lossless widening;
- SwiGLU: computed in fp32, rounded to fp16 once, as down_proj's input cast rounded it;
- down output: fp16, and `residual (fp32) + mlp (fp16)` promotes to fp32 with the same values.
Backward: the two separate casts (`xa`, `xb`) keep the gate/up input gradients flowing into
fp32 separately and summing in fp32, exactly as through bitsandbytes' two internal casts.
"""

from __future__ import annotations

import functools

import torch

ENABLED = True


def _is_fp16_linear4bit(module) -> bool:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    return (isinstance(module, bnb.nn.Linear4bit)
            and getattr(module, "compute_type_is_set", False)
            and module.compute_dtype == torch.float16)


def patch_qwen2_mlp_fp16():
    """Wraps the installed Qwen2MLP.forward; install before patch_qwen2_unpad()."""
    from transformers.models.qwen2 import modeling_qwen2

    from .swiglu import swiglu

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls, "_soup_mlp_fp16", False):
        return
    original = cls.forward

    @functools.wraps(original)
    def forward(self, x):
        if not (ENABLED and x.is_cuda and x.dtype == torch.float32
                and getattr(self.config, "hidden_act", "silu") == "silu"
                and _is_fp16_linear4bit(self.gate_proj) and _is_fp16_linear4bit(self.up_proj)
                and _is_fp16_linear4bit(self.down_proj)):
            return original(self, x)
        xa = x.to(torch.float16)  # two casts on purpose: see the module docstring
        xb = x.to(torch.float16)
        return self.down_proj(swiglu(self.gate_proj(xa), self.up_proj(xb)))

    cls.forward = forward
    cls._soup_mlp_fp16 = True
