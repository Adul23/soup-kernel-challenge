"""The fused head + loss when the tied fp16 head weight lives in pinned host RAM (offload.py).

The vocabulary is the outer loop: the weight comes to the GPU in VCHUNK-row pieces (two GPU
slots, the next piece copied on a side stream while the current one is used), and every piece
is used for all token rows at once, so the 467 MB matrix crosses PCIe twice per step instead of
living on the GPU.

One pass over the vocabulary does everything. For each piece, cuBLAS computes the logits of all
real rows (fp16 in, fp32 accumulate, fp16 out, as autocast would), and `head_chunk_stats_kernel`
carries each row's online max / sum-exp / entropy sum / first-index argmax / target logit across
pieces in fp32 arrays (VCHUNK is a multiple of BLOCK, so the blocks are visited in the same order
with the same arithmetic as `fused_head_ce_kernel` in head.py; the last piece writes loss,
entropy, correct and the log-sum-exp). For labelled rows the kernel also overwrites the piece's
logits with GRAD_SCALE * exp(z - m) / d, normalised by the running max m and sum-exp d (a row
sums to at most GRAD_SCALE, so the fp16 product cannot overflow), and returns the factor
exp(m_before - m) * d_before / d; cuBLAS then accumulates acc = acc * factor + P @ W_piece in
fp32, as flash attention accumulates its output. At the end acc = GRAD_SCALE * softmax @ W, and
the hidden-state gradient is GRAD_SCALE * (softmax @ W - W[target]). So the weight crosses PCIe once per step and
the logits are computed once, instead of a second pass for the gradient.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

VCHUNK = 4096  # vocabulary rows per piece: 4096 x 1536 x 2 B = 12 MB per slot
BLOCK = 4096  # as head.BLOCK: VCHUNK must be a multiple of it
IGNORE_INDEX = -100
GRAD_SCALE = 256.0


@triton.jit(do_not_specialize=["v0", "VC", "FIRST", "LAST", "KLAB"])
def head_chunk_stats_kernel(
    LOGITS,
    stride_row,
    TARGET,
    M,
    D,
    S,
    BEST,
    ARG,
    ZT,
    LOSS,
    ENTROPY,
    CORRECT,
    LSE,
    v0,
    VC,
    FIRST,
    LAST,
    ALPHA,
    KLAB,
    IGNORE: tl.constexpr,
    GRAD_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ptr = LOGITS + row * stride_row
    tgt = tl.load(TARGET + row)
    first = FIRST != 0  # runtime flags (no branch: one compiled variant)
    m = tl.where(first, float("-inf"), tl.load(M + row))
    d = tl.where(first, 0.0, tl.load(D + row))
    s = tl.where(first, 0.0, tl.load(S + row))
    zt = tl.where(first, 0.0, tl.load(ZT + row))
    best = tl.where(first, float("-inf"), tl.load(BEST + row))
    arg = tl.where(first, 0, tl.load(ARG + row))
    for start in range(0, VC, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        msk = offs < VC
        z = tl.load(ptr + offs, mask=msk, other=float("-inf")).to(tl.float32)
        bmax = tl.max(z, axis=0)
        barg = tl.argmax(z, axis=0)
        m_new = tl.maximum(m, bmax)
        alpha = tl.exp(m - m_new)
        e = tl.where(msk, tl.exp(z - m_new), 0.0)
        d = d * alpha + tl.sum(e, axis=0)
        s = s * alpha + tl.sum(tl.where(msk, e * z, 0.0), axis=0)
        m = m_new
        take = bmax > best
        arg = tl.where(take, barg + start + v0, arg)
        best = tl.maximum(best, bmax)
        zt += tl.sum(tl.where(offs + v0 == tgt, z, 0.0), axis=0)
    # One pass for the gradient too (as flash attention accumulates its output): labelled rows
    # get P = GRAD_SCALE * exp(z - m) / d for this piece, normalised by the running max m and
    # sum-exp d, so a row of P sums to at most GRAD_SCALE and P @ W cannot overflow fp16 (as
    # with the final softmax); ALPHA = exp(m_before - m) * d_before / d rescales what was
    # accumulated from the earlier pieces (0 for the first piece).
    m_before = tl.where(first, float("-inf"), tl.load(M + row))
    d_before = tl.where(first, 0.0, tl.load(D + row))
    lab_row = row < KLAB
    tl.store(ALPHA + row, tl.exp(m_before - m) * (d_before / d), mask=lab_row)
    scale = GRAD_SCALE / d
    for start in range(0, VC, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        msk = (offs < VC) & lab_row
        z = tl.load(ptr + offs, mask=msk, other=0.0).to(tl.float32)
        p = tl.exp(z - m) * scale
        tl.store(ptr + offs, p.to(LOGITS.dtype.element_ty), mask=msk)
    last = LAST != 0
    lse = m + tl.log(d)
    labelled = tgt != IGNORE
    tl.store(LOSS + row, tl.where(labelled, lse - zt, 0.0), mask=last)
    tl.store(ENTROPY + row, lse - s / d, mask=last)
    tl.store(CORRECT + row, ((arg == tgt) & labelled).to(tl.int32), mask=last)
    tl.store(LSE + row, lse, mask=last)
    tl.store(M + row, m)
    tl.store(D + row, d)
    tl.store(S + row, s)
    tl.store(ZT + row, zt)
    tl.store(BEST + row, best)
    tl.store(ARG + row, arg)


class _Pieces:
    """Pieces of a pinned host matrix, double-buffered on the GPU."""

    def __init__(self, host, dev):
        self.host, self.dev = host, dev
        self.n = (host.shape[0] + VCHUNK - 1) // VCHUNK
        self.bufs = [
            torch.empty((VCHUNK, host.shape[1]), dtype=host.dtype, device=dev)
            for _ in range(2)
        ]
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.free = [torch.cuda.Event(), torch.cuda.Event()]
        self.stream = torch.cuda.Stream(device=dev)

    def _issue(self, j):
        slot = j % 2
        v0 = j * VCHUNK
        rows = min(VCHUNK, self.host.shape[0] - v0)
        self.stream.wait_event(self.free[slot])
        with torch.cuda.stream(self.stream):
            self.bufs[slot][:rows].copy_(self.host[v0 : v0 + rows], non_blocking=True)
            self.ready[slot].record(self.stream)

    def __iter__(self):
        self._issue(0)
        for j in range(self.n):
            if j + 1 < self.n:
                self._issue(j + 1)
            slot = j % 2
            torch.cuda.current_stream().wait_event(self.ready[slot])
            v0 = j * VCHUNK
            rows = min(VCHUNK, self.host.shape[0] - v0)
            yield j, v0, self.bufs[slot][:rows]
            self.free[slot].record(torch.cuda.current_stream())


_WARM = {"thread": None}


def _warmup():
    dev = torch.device("cuda", torch.cuda.current_device())
    logits = torch.empty((1, VCHUNK), dtype=torch.float16, device=dev)
    t = torch.zeros(1, dtype=torch.long, device=dev)
    f = [torch.empty(1, dtype=torch.float32, device=dev) for _ in range(9)]
    i = [torch.empty(1, dtype=torch.int32, device=dev) for _ in range(2)]
    head_chunk_stats_kernel.warmup(
        logits,
        VCHUNK,
        t,
        f[0],
        f[1],
        f[2],
        f[3],
        i[0],
        f[4],
        f[5],
        f[6],
        i[1],
        f[7],
        0,
        VCHUNK,
        1,
        0,
        f[8],
        1,
        IGNORE=IGNORE_INDEX,
        GRAD_SCALE=GRAD_SCALE,
        BLOCK=BLOCK,
        num_warps=8,
        grid=(1,),
    )


def start_warmup():
    """Compile both kernels in a thread while Soup loads the model (joined at first use)."""
    import threading

    if _WARM["thread"] is None and torch.cuda.is_available():

        def run():
            try:
                _warmup()
            except Exception:  # noqa: BLE001 - then they compile at first use, as usual
                pass

        _WARM["thread"] = threading.Thread(target=run, daemon=True)
        _WARM["thread"].start()


def head_streamed(h, weight_host, tgt, rows_lab, rows_ent, need_grad):
    """h [n, dim] fp32 on the GPU; returns token_loss, token_entropy, correct, grad (or None)
    exactly as head.FusedHeadLoss.forward computes them."""
    if _WARM["thread"] is not None:
        _WARM["thread"].join()
    dev = h.device
    n, dim = h.shape
    rows = torch.cat([rows_lab, rows_ent])
    k, k_lab = rows.numel(), rows_lab.numel()
    token_loss = torch.zeros(n, dtype=torch.float32, device=dev)
    token_entropy = torch.zeros(n, dtype=torch.float32, device=dev)
    grad = torch.zeros((n, dim), dtype=torch.float32, device=dev) if need_grad else None
    correct = torch.zeros((), dtype=torch.long, device=dev)
    if k == 0:
        return token_loss, token_entropy, correct, grad

    x = h.index_select(0, rows).to(torch.float16)
    t = tgt.index_select(0, rows).contiguous()
    f32 = dict(dtype=torch.float32, device=dev)
    m, d, s, zt, best = (torch.empty(k, **f32) for _ in range(5))
    arg = torch.empty(k, dtype=torch.int32, device=dev)
    loss_k, ent_k, lse = (torch.empty(k, **f32) for _ in range(3))
    cor_k = torch.empty(k, dtype=torch.int32, device=dev)
    logits = torch.empty((k, VCHUNK), dtype=torch.float16, device=dev)

    want_grad = bool(need_grad and k_lab)
    alpha = torch.empty(k, **f32)
    acc = torch.zeros((k_lab, dim), **f32) if want_grad else None

    pieces = _Pieces(weight_host, dev)
    for (
        j,
        v0,
        w,
    ) in pieces:  # one pass: statistics and gradient over the whole vocabulary
        vc = w.shape[0]
        out = logits[:, :vc]
        torch.mm(x, w.t(), out=out)
        head_chunk_stats_kernel[(k,)](
            out,
            logits.stride(0),
            t,
            m,
            d,
            s,
            best,
            arg,
            zt,
            loss_k,
            ent_k,
            cor_k,
            lse,
            v0,
            vc,
            int(j == 0),
            int(j == pieces.n - 1),
            alpha,
            k_lab if want_grad else 0,
            IGNORE=IGNORE_INDEX,
            GRAD_SCALE=GRAD_SCALE,
            BLOCK=BLOCK,
            num_warps=8,
        )
        if want_grad:  # acc = acc * alpha + P @ W_piece, fp32 accumulation (fp16 products)
            acc.mul_(alpha[:k_lab, None]).add_(torch.mm(out[:k_lab], w))
    token_loss.index_copy_(0, rows, loss_k)
    token_entropy.index_copy_(0, rows, ent_k)
    correct += cor_k.sum()

    if want_grad:
        # acc is GRAD_SCALE * softmax @ W now (already normalised); GRAD_SCALE * (softmax @ W -
        # W[target]) for the labelled rows. The caller divides by denominator * GRAD_SCALE, as
        # for the resident head.
        t_cpu = t[:k_lab].to("cpu")
        w_tgt = weight_host.index_select(0, t_cpu).to(dev, non_blocking=True)
        acc.sub_(w_tgt.float().mul_(GRAD_SCALE))
        grad.index_copy_(0, rows_lab, acc)
    return token_loss, token_entropy, correct, grad
