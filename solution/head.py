"""Fused output head + causal-LM loss: no [batch, seq, 151936] logits tensor in training.

Base Soup on a T4 (fp16 autocast): the tied fp32 head weight is cast to fp16 on every forward, the
fp16 logits [B, T, V] are upcast to fp32 by ForCausalLMLoss, TRL reads them again for entropy and
token accuracy, Soup's rewind recorder reads them a third time for per-row losses, and backward
keeps fp32 and fp16 gradients of the same shape. With V = 151 936 that is gigabytes at the peak.

Here, during training only:
- cuBLAS computes the logits of up to CHUNK tokens at a time into one reused fp16 buffer, exactly
  as autocast would (fp16 inputs, fp32 accumulation, fp16 output);
- our Triton kernel `fused_head_ce_kernel` reads each row once and computes, in fp32, an online
  max / sum-exp (log-sum-exp), the entropy, the first-index argmax and the target logit; for
  labelled rows a second pass writes d(loss)/d(logits) = softmax - onehot back into the buffer,
  scaled by GRAD_SCALE so small probabilities do not flush to zero in fp16;
- cuBLAS multiplies that gradient by the head weight to get the hidden-state gradient, which is
  kept (fp32, [B*T, hidden], ~12 MB) and handed to autograd in backward.
Pad positions are skipped entirely; prompt positions only need entropy (no gradient pass).

The trainer side keeps every Soup output: TRL's `entropy`, `num_tokens`, `mean_token_accuracy`
are logged by TRL's own formulas from the kernel's per-token results, and Soup's rewind recorder
reads per-token losses through its own no-logits contract (`_soup_chunked_step`, Soup #1693).

Ideas: chunked fused linear cross-entropy (Liger Kernel, arXiv:2410.10989) and online softmax
statistics (Milakov & Gimelshein 2018; Cut Your Losses, arXiv:2411.09009). The code is our own.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
CHUNK = 128  # tokens per logits buffer: 128 x 151 936 x 2 B = 39 MB
BLOCK = 4096  # vocabulary entries per kernel iteration
GRAD_SCALE = 256.0  # fp16 headroom for softmax - onehot; divided out in fp32 afterwards
ENABLED = True  # tests flip this to get the reference path

try:  # the attribute Soup's rewind recorder reads when a step returns no logits
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR
except Exception:  # noqa: BLE001 - Soup not importable (e.g. a bare unit test)
    STEP_ATTR = "_soup_chunked_step"


@triton.jit
def fused_head_ce_kernel(
    LOGITS,
    stride_row,
    TARGET,
    LOSS,
    ENTROPY,
    CORRECT,
    V,
    IGNORE: tl.constexpr,
    WRITE_GRAD: tl.constexpr,
    GRAD_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token row of the logits buffer."""
    row = tl.program_id(0).to(tl.int64)
    ptr = LOGITS + row * stride_row
    tgt = tl.load(TARGET + row)

    m = float("-inf")  # running max
    d = 0.0  # running sum exp(z - m)
    s = 0.0  # running sum exp(z - m) * z, for the entropy
    zt = 0.0  # logit of the target
    best = float("-inf")
    arg = 0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        msk = offs < V
        z = tl.load(ptr + offs, mask=msk, other=float("-inf")).to(tl.float32)
        bmax = tl.max(z, axis=0)
        barg = tl.argmax(z, axis=0)  # first index among equal maxima
        m_new = tl.maximum(m, bmax)
        alpha = tl.exp(m - m_new)
        e = tl.where(msk, tl.exp(z - m_new), 0.0)
        d = d * alpha + tl.sum(e, axis=0)
        s = s * alpha + tl.sum(tl.where(msk, e * z, 0.0), axis=0)
        m = m_new
        take = bmax > best  # strict: an earlier block keeps a tie, as torch.argmax does
        arg = tl.where(take, barg + start, arg)
        best = tl.maximum(best, bmax)
        zt += tl.sum(tl.where(offs == tgt, z, 0.0), axis=0)

    lse = m + tl.log(d)
    labelled = tgt != IGNORE
    tl.store(LOSS + row, tl.where(labelled, lse - zt, 0.0))
    tl.store(ENTROPY + row, lse - s / d)  # = -sum p log p
    tl.store(CORRECT + row, ((arg == tgt) & labelled).to(tl.int32))

    if WRITE_GRAD:  # in place: the row's logits become GRAD_SCALE * (softmax - onehot)
        for start in range(0, V, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            msk = offs < V
            z = tl.load(ptr + offs, mask=msk, other=0.0).to(tl.float32)
            p = tl.exp(z - lse)
            g = tl.where(offs == tgt, p - 1.0, p) * GRAD_SCALE
            tl.store(ptr + offs, g.to(LOGITS.dtype.element_ty), mask=msk)


_W16: dict = {}


def fp16_weight(weight: torch.Tensor) -> torch.Tensor:
    """The frozen head weight in fp16, cast once (autocast recasts it on every forward)."""
    if weight.dtype == torch.float16:
        return weight.detach()
    key = (weight.data_ptr(), weight._version, tuple(weight.shape), str(weight.device))
    if _W16.get("key") != key:
        _W16.pop("w", None)
        _W16["w"] = weight.detach().to(torch.float16).contiguous()
        _W16["key"] = key
    return _W16["w"]


class FusedHeadLoss(torch.autograd.Function):
    """loss = sum_t CE(fp16(h_t) @ fp16(W).T, target_t) / denominator, plus per-token metrics."""

    @staticmethod
    def forward(ctx, hidden, weight16, target, real, denominator, chunk):
        b, t, dim = hidden.shape
        v = weight16.shape[0]
        n = b * t
        dev = hidden.device
        need_grad = ctx.needs_input_grad[0]
        with torch.autocast(device_type=dev.type, enabled=False):
            h = hidden.detach().reshape(n, dim)
            tgt = target.reshape(n).to(device=dev, dtype=torch.long)
            lab = tgt != IGNORE_INDEX
            rows_lab = lab.nonzero().squeeze(1)
            rows_ent = (
                (real.reshape(n).to(dev) & ~lab).nonzero().squeeze(1)
            )  # prompt tokens

            token_loss = torch.zeros(n, dtype=torch.float32, device=dev)
            token_entropy = torch.zeros(n, dtype=torch.float32, device=dev)
            correct = torch.zeros((), dtype=torch.long, device=dev)
            grad = (
                torch.zeros((n, dim), dtype=torch.float32, device=dev)
                if need_grad
                else None
            )

            rows_max = max(rows_lab.numel(), rows_ent.numel(), 1)
            streamed = (
                not weight16.is_cuda
            )  # tied weight in pinned host RAM (offload.py)
            if streamed:
                from .head_stream import head_streamed

                token_loss, token_entropy, correct, grad = head_streamed(
                    h, weight16, tgt, rows_lab, rows_ent, need_grad
                )
            else:
                buf = torch.empty(
                    (min(chunk, rows_max), v), dtype=torch.float16, device=dev
                )
            for rows, write_grad in (
                () if streamed else ((rows_lab, need_grad), (rows_ent, False))
            ):
                for s0 in range(0, rows.numel(), chunk):
                    r = rows[s0 : s0 + chunk]
                    c = r.numel()
                    logits = buf[:c]
                    torch.mm(
                        h.index_select(0, r).to(torch.float16), weight16.t(), out=logits
                    )
                    t_c = tgt.index_select(0, r).contiguous()
                    loss_c = torch.empty(c, dtype=torch.float32, device=dev)
                    ent_c = torch.empty(c, dtype=torch.float32, device=dev)
                    cor_c = torch.empty(c, dtype=torch.int32, device=dev)
                    fused_head_ce_kernel[(c,)](
                        logits,
                        logits.stride(0),
                        t_c,
                        loss_c,
                        ent_c,
                        cor_c,
                        v,
                        IGNORE=IGNORE_INDEX,
                        WRITE_GRAD=bool(write_grad),
                        GRAD_SCALE=GRAD_SCALE,
                        BLOCK=BLOCK,
                        num_warps=8,
                    )
                    token_loss.index_copy_(0, r, loss_c)
                    token_entropy.index_copy_(0, r, ent_c)
                    correct += cor_c.sum()
                    if write_grad:
                        grad.index_copy_(0, r, torch.mm(logits, weight16).float())

            loss = token_loss.sum() / denominator
            if grad is not None:
                grad.div_(denominator * GRAD_SCALE)

        ctx.hidden_shape, ctx.hidden_dtype = hidden.shape, hidden.dtype
        ctx.save_for_backward(grad if grad is not None else hidden.new_empty(0))
        ctx.has_grad = grad is not None
        token_entropy, token_loss = token_entropy.view(b, t), token_loss.view(b, t)
        ctx.mark_non_differentiable(token_entropy, token_loss, correct)
        return loss, token_entropy, token_loss, correct

    @staticmethod
    def backward(ctx, grad_loss, _g_entropy, _g_token_loss, _g_correct):
        if not ctx.has_grad:
            return None, None, None, None, None, None
        (grad,) = ctx.saved_tensors
        grad_hidden = (grad * grad_loss).view(ctx.hidden_shape).to(ctx.hidden_dtype)
        return grad_hidden, None, None, None, None, None


def fused_head_loss(
    hidden, weight, labels, attention_mask=None, num_items_in_batch=None, chunk=CHUNK
):
    """ForCausalLMLoss(fp16 head(hidden), labels) and the metrics, without the full logits."""
    target = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    target = target.to(hidden.device)
    if (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.shape == target.shape
    ):
        real = attention_mask.to(hidden.device) != 0
    else:
        real = torch.ones_like(target, dtype=torch.bool)
    labelled = (target != IGNORE_INDEX).sum()
    if (
        num_items_in_batch is None
    ):  # as fixed_cross_entropy: "mean" over labelled tokens
        denominator = labelled
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = num_items_in_batch
    loss, entropy, token_loss, correct = FusedHeadLoss.apply(
        hidden, fp16_weight(weight), target, real, denominator, chunk
    )
    return loss, HeadStep(
        token_entropy=entropy, token_loss=token_loss, correct=correct, labelled=labelled
    )


@dataclass(frozen=True)
class HeadStep:
    """What Soup's rewind recorder and TRL's metrics need from a step without logits."""

    token_entropy: Any  # [B, T] fp32, entropy of each position (0 at pads)
    token_loss: Any  # [B, T] fp32, CE of label t+1 at position t, 0 where unlabelled
    correct: Any  # labelled positions whose argmax is the label
    labelled: Any  # labelled positions


class _Context:
    armed = False  # set by the trainer patch around its forward
    step = None  # HeadStep left by the model patch


_CTX = _Context()


def _autocast_fp16() -> bool:
    try:
        return (
            torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
        )
    except TypeError:  # older torch
        return (
            torch.is_autocast_enabled()
            and torch.get_autocast_gpu_dtype() == torch.float16
        )


def _model_can_engage(model, args, kwargs) -> bool:
    if not (ENABLED and _CTX.armed and model.training) or args:
        return False
    labels = kwargs.get("labels")
    if not isinstance(labels, torch.Tensor) or labels.dim() != 2 or not labels.is_cuda:
        return False
    if (
        kwargs.get("shift_labels") is not None
        or kwargs.get("past_key_values") is not None
    ):
        return False
    if kwargs.get("use_cache") or kwargs.get("return_dict") is False:
        return False
    if kwargs.get("logits_to_keep", 0) not in (0, None):
        return False
    if getattr(model.config, "final_logit_softcapping", None) is not None:
        return False
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dim() != 2
        or weight.requires_grad
    ):
        return False
    if (
        weight.dtype not in (torch.float16, torch.float32)
        or getattr(head, "bias", None) is not None
    ):
        return False
    if weight.shape[0] != model.config.vocab_size:
        return False
    return _autocast_fp16()


def _fp16_exact(weight: torch.Tensor, rows: int = 8192) -> bool:
    """True when every value of the fp32 weight survives fp32 -> fp16 -> fp32 unchanged
    (checked in row chunks, so no full-size temporary)."""
    with torch.no_grad():
        for start in range(0, weight.shape[0], rows):
            part = weight[start : start + rows]
            if not torch.equal(part.half().float(), part):
                return False
    return True


def _upcast_output(module, inputs, output):
    return output.float()


def compact_tied_embedding(model) -> bool:
    """Store the frozen tied embedding / head matrix in fp16 instead of fp32, once.

    prepare_model_for_kbit_training widens it to fp32 (0.93 GB), and the head kept a second,
    fp16 copy (0.47 GB). When the fp32 values are all fp16 values (the model is loaded in fp16 on
    a T4), one fp16 tensor holds the same numbers: the head uses it directly, and the embedding
    output is widened back to fp32 by a forward hook, so the residual stream keeps its dtype and
    every value. Anything else (untied, trainable, not fp16-exact) is left as it is."""
    if getattr(model, "_soup_embedding_checked", False):
        return False
    model._soup_embedding_checked = True
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    weight = getattr(embed, "weight", None)
    if (
        head is None
        or weight is None
        or head.weight is not weight
        or weight.requires_grad
        or weight.dtype != torch.float32
        or not weight.is_cuda
        or not isinstance(embed, torch.nn.Embedding)
    ):
        return False
    if not _fp16_exact(weight):
        print(
            "[solution.head] tied embedding is not fp16-exact: kept in fp32", flush=True
        )
        return False
    _W16.clear()  # the separate fp16 copy of the head goes away with this
    weight.data = weight.data.to(torch.float16)
    embed.register_forward_hook(_upcast_output)
    print(
        "[solution.head] tied embedding stored in fp16 (values unchanged)", flush=True
    )
    return True


def patch_qwen2_fused_head():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls, "_soup_fused_head", False):
        return
    original = cls.forward

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        if not _model_can_engage(self, args, kwargs):
            return original(self, *args, **kwargs)
        compact_tied_embedding(self)
        labels = kwargs.pop("labels")
        kwargs.pop("logits_to_keep", None)
        kwargs.pop("return_dict", None)
        # Qwen2ForCausalLM.forward of transformers 5.16.1 up to the head; loss kwargs such as
        # num_items_in_batch go to the decoder as they do there.
        outputs = self.model(**kwargs)
        loss, step = fused_head_loss(
            outputs.last_hidden_state,
            self.lm_head.weight,
            labels,
            attention_mask=kwargs.get("attention_mask"),
            num_items_in_batch=kwargs.get("num_items_in_batch"),
        )
        _CTX.step = step
        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    cls.forward = forward
    cls._soup_fused_head = True


def _trainer_can_engage(trainer, inputs) -> bool:
    if not ENABLED or not getattr(getattr(trainer, "model", None), "training", False):
        return False  # training steps only; eval / predict keep TRL's path
    if (
        "labels" not in inputs
        or "shift_labels" in inputs
        or "attention_mask" not in inputs
    ):
        return False
    args = getattr(trainer, "args", None)
    if getattr(args, "use_liger_kernel", False) or getattr(args, "padding_free", False):
        return False
    if getattr(trainer, "num_virtual_tokens", 0) or getattr(
        trainer, "aux_loss_enabled", False
    ):
        return False
    if getattr(trainer, "compute_loss_func", None) is not None:
        return False
    return getattr(trainer, "label_smoother", None) is None and hasattr(
        trainer, "_metrics"
    )


def log_step_metrics(trainer, mode, inputs, step: HeadStep) -> None:
    """TRL 0.29.1 SFTTrainer.compute_loss metrics, same names, order and formulas."""
    acc = trainer.accelerator
    attention_mask = inputs["attention_mask"]
    entropy = torch.sum(step.token_entropy * attention_mask) / attention_mask.sum()
    trainer._metrics[mode]["entropy"].append(
        acc.gather_for_metrics(entropy).mean().item()
    )
    if mode == "train":
        trainer._total_train_tokens += (
            acc.gather_for_metrics(attention_mask.sum()).sum().item()
        )
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    correct = acc.gather_for_metrics(step.correct)
    total = acc.gather_for_metrics(step.labelled)
    total_sum = total.sum()
    accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
    trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def log_logit_metrics(trainer, mode, inputs, labels, logits) -> None:
    """The same metrics from logits, for a step the model patch did not take (fallback)."""
    from trl.trainer.utils import entropy_from_logits

    with torch.no_grad():
        attention_mask = inputs["attention_mask"]
        target = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
        per_token = entropy_from_logits(logits)
        labelled = target != IGNORE_INDEX
        step = HeadStep(
            token_entropy=per_token,
            token_loss=None,
            correct=((logits.argmax(-1) == target) & labelled).sum(),
            labelled=labelled.sum(),
        )
        log_step_metrics(trainer, mode, inputs, step)


def patch_sft_compute_loss():
    from trl.trainer import sft_trainer

    cls = sft_trainer.SFTTrainer
    if getattr(cls, "_soup_fused_head", False):
        return
    original = cls.compute_loss

    @functools.wraps(original)
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        from . import warmup

        warmup.wait()  # kernels compiled during model loading (no-op after the first step)
        if not _trainer_can_engage(self, inputs):
            return original(
                self,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)  # as TRL does before it delegates
        inputs["use_cache"] = False
        labels = inputs["labels"]
        _CTX.armed, _CTX.step = True, None
        try:
            loss, outputs = super(cls, self).compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            _CTX.armed = False
        step, _CTX.step = _CTX.step, None
        setattr(
            self, STEP_ATTR, step
        )  # rewind reads step.token_loss when logits are None
        if step is not None:
            log_step_metrics(self, mode, inputs, step)
        else:
            log_logit_metrics(self, mode, inputs, labels, outputs.logits)
        return (loss, outputs) if return_outputs else loss

    cls.compute_loss = compute_loss
    cls._soup_fused_head = True


def patch_fused_head():
    """Both halves or neither: the model patch only runs when the trainer patch arms it."""
    try:
        patch_sft_compute_loss()
    except ImportError:  # no trl: nothing would arm the model patch
        return
    patch_qwen2_fused_head()
