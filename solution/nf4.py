"""NF4 weight dequantization in Triton for bitsandbytes 4-bit linears (forward and backward).

Base Soup: every Linear4bit (7 per layer x 28 layers) dequantizes its NF4 weight to fp16 twice
per step, once in forward (bitsandbytes' gemm_4bit falls back to dequantize + F.linear for these
shapes) and once in backward (MatMul4Bit.backward). That is bitsandbytes' `kDequantizeBlockwise`,
about 5% of the GPU time in the base profile, plus a second kernel each time that rebuilds the
fp32 absmax from its 8-bit double quantization.

Here `nf4_dequant_kernel` writes the fp16 weight straight from the packed bytes. The fp32 absmax of
each frozen weight is rebuilt once, by bitsandbytes itself, and cached (about 80 MB for the whole
model), so it is no longer recomputed 392 times a step. The arithmetic is the one in
bitsandbytes' csrc/kernels.cu: element 2i is LUT[byte_i >> 4] * absmax, element 2i+1 is
LUT[byte_i & 15] * absmax, one fp32 multiply, rounded to fp16 on store; the output is bit-identical
(test_nf4.py checks it with torch.equal). The matmuls stay in cuBLAS, as in base Soup.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

ENABLED = True
BLOCK = 1024  # packed bytes per program -> 2048 weights

# bitsandbytes csrc/kernels.cu nf4_dequantization_lut (fp32 literals).
NF4_LUT = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


@triton.jit
def nf4_dequant_kernel(Q, ABSMAX, LUT, OUT, n_bytes,
                       HALF_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    ok = offs < n_bytes
    q = tl.load(Q + offs, mask=ok, other=0).to(tl.int32)
    scale = tl.load(ABSMAX + offs // HALF_BLOCK, mask=ok, other=0.0)  # one absmax per block
    hi = tl.load(LUT + (q >> 4)) * scale
    lo = tl.load(LUT + (q & 15)) * scale
    vals = tl.reshape(tl.join(hi, lo), (2 * BLOCK,))  # hi0, lo0, hi1, lo1, ...
    out = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(OUT + out, vals.to(OUT.dtype.element_ty), mask=out < 2 * n_bytes)


_LUT: dict = {}
_ABSMAX: dict = {}


def _lut(device):
    lut = _LUT.get(device)
    if lut is None:
        lut = _LUT[device] = torch.tensor(NF4_LUT, dtype=torch.float32, device=device)
    return lut


# offload.py: fn(quant_state) -> the fp32 absmax on the GPU (streamed with the layer), or None.
ABSMAX_SOURCE = None


def build_absmax_fp32(quant_state):
    """The fp32 absmax bitsandbytes rebuilds on every call, rebuilt here by bitsandbytes."""
    import bitsandbytes.functional as bnbf

    if quant_state.nested:
        absmax = bnbf.dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax = absmax + quant_state.offset
    else:
        absmax = quant_state.absmax
    return absmax.float().contiguous()


def absmax_fp32(quant_state):
    """build_absmax_fp32(), once per weight: cached here unless offload.py streams it."""
    if ABSMAX_SOURCE is not None:
        absmax = ABSMAX_SOURCE(quant_state)
        if absmax is not None:
            return absmax
    hit = _ABSMAX.get(id(quant_state))
    if hit is not None and hit[0] is quant_state:
        return hit[1]
    absmax = build_absmax_fp32(quant_state)
    _ABSMAX[id(quant_state)] = (quant_state, absmax)
    return absmax


class _Launch:
    """Everything a dequant launch of one frozen weight needs except the output buffer and the
    absmax (which offload.py may stream), built on the first call and reused (389 launches a
    step go through this)."""

    __slots__ = ("state", "q", "lut", "n_bytes", "grid", "half", "shape")

    def __init__(self, packed, quant_state):
        self.state = quant_state
        self.q = packed.data.reshape(-1)  # a plain tensor even when `packed` is a Params4bit
        self.lut = _lut(self.q.device)
        self.n_bytes = self.q.numel()
        self.grid = (triton.cdiv(self.n_bytes, BLOCK),)
        self.half = quant_state.blocksize // 2
        self.shape = tuple(quant_state.shape)


_LAUNCH: dict = {}


def _same_weight(q, packed) -> bool:
    """`packed` is a fresh .t() view on every Linear4bit call, so compare storage, not objects.
    The cached view keeps that storage alive, so its address cannot pass to another weight."""
    return (q.data_ptr() == packed.data_ptr() and q.numel() == packed.numel()
            and q.dtype == packed.dtype and q.device == packed.device)


def _launch(packed, quant_state):
    hit = _LAUNCH.get(id(quant_state))
    if hit is None or hit.state is not quant_state or not _same_weight(hit.q, packed):
        hit = _LAUNCH[id(quant_state)] = _Launch(packed, quant_state)
    return hit


def dequantize_nf4(packed, quant_state, dtype=torch.float16):
    """[N, K] weight of a 4-bit NF4 tensor, as bitsandbytes.functional.dequantize_4bit."""
    ln = _launch(packed, quant_state)
    out = torch.empty(ln.shape, dtype=dtype, device=ln.q.device)
    nf4_dequant_kernel[ln.grid](
        ln.q, absmax_fp32(quant_state), ln.lut, out, ln.n_bytes,
        HALF_BLOCK=ln.half, BLOCK=BLOCK, num_warps=4,
    )
    return out


class NF4Linear(torch.autograd.Function):
    """F.linear(A, dequant(B), bias) with the weight dequantized again in backward, as
    bitsandbytes' MatMul4Bit does (nothing of the weight is kept between the two)."""

    @staticmethod
    def forward(ctx, A, B, bias, quant_state):
        ctx.state, ctx.B = quant_state, B
        weight = dequantize_nf4(B, quant_state, A.dtype)
        return torch.nn.functional.linear(A, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_a = grad_bias = None
        if ctx.needs_input_grad[0]:
            weight = dequantize_nf4(ctx.B, ctx.state, grad_output.dtype)
            grad_a = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_a, None, grad_bias, None


def weight_ok(B, quant_state) -> bool:
    """A packed NF4 weight this kernel dequantizes exactly as bitsandbytes does."""
    if not ENABLED or quant_state is None or not isinstance(B, torch.Tensor):
        return False
    if getattr(quant_state, "quant_type", None) != "nf4" or len(quant_state.shape) != 2:
        return False
    if quant_state.dtype not in (torch.float16, torch.float32):
        return False  # bf16 would round twice in bitsandbytes' backward; keep its path
    n, k = quant_state.shape
    if (n * k) % 2 or quant_state.blocksize % 2:
        return False
    if B.dtype != torch.uint8 or B.numel() * 2 != n * k or not B.is_cuda:
        return False
    return not quant_state.nested or getattr(quant_state.state2, "blocksize", None) == 256


_OK: dict = {}


def _weight_ok_cached(B, quant_state) -> bool:
    """weight_ok() of a frozen weight, evaluated once per (weight, quant_state) pair."""
    if not ENABLED:
        return False
    hit = _OK.get(id(quant_state))
    if (hit is None or hit[0] is not quant_state or not isinstance(B, torch.Tensor)
            or not _same_weight(hit[1], B)):
        ok = weight_ok(B, quant_state)
        if not isinstance(B, torch.Tensor):
            return ok
        hit = _OK[id(quant_state)] = (quant_state, B.data.reshape(-1), ok)
    return hit[2]


def _usable(A, B, quant_state, out, bias) -> bool:
    if out is not None or not _weight_ok_cached(B, quant_state):
        return False
    if not (isinstance(A, torch.Tensor) and A.is_cuda and A.dtype == torch.float16 and A.numel()):
        return False
    if A.shape[-1] != quant_state.shape[1]:
        return False
    return bias is None or bias.dtype == A.dtype


def patch_bnb_matmul_4bit():
    """Linear4bit.forward calls `bnb.matmul_4bit` by attribute, so the module attribute is the
    hook; anything outside the case above goes to bitsandbytes unchanged."""
    import bitsandbytes as bnb

    if getattr(bnb, "_soup_nf4", False):
        return
    original = bnb.matmul_4bit

    @functools.wraps(original)
    def matmul_4bit(A, B, quant_state=None, out=None, bias=None):
        if not _usable(A, B, quant_state, out, bias):
            return original(A, B, quant_state=quant_state, out=out, bias=bias)
        return NF4Linear.apply(A, B, bias, quant_state)

    bnb.matmul_4bit = matmul_4bit
    bnb._soup_nf4 = True
