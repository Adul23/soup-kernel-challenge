"""Qwen2 decoder layers on the real tokens only, attention included.

unpad.py already skips the pads in the MLP. Everything else in a decoder layer still ran on all
B*T positions: both RMSNorms, both residual additions, the q/k/v/o projections with their LoRA,
the fp32<->fp16 casts around them, the rotary embedding and attention itself. Here the residual
stream stays packed as [n_real, hidden] from the first layer to the final norm (unpad.py scatters
it back to [B, T, hidden] when Qwen2Model returns), and a layer runs on the packed rows:

    input_layernorm -> q/k/v (+LoRA) -> rotary -> varlen attention -> o_proj -> + residual
    post_attention_layernorm -> MLP -> + residual

Attention runs on the packed tokens with PyTorch's memory-efficient attention kernel (the one SDPA
already uses on the T4) given cu_seqlens, causal within each row: no padded queries or keys at
all. It needs right padding (each row's real tokens a prefix); otherwise, or if the op is
unavailable, q/k/v are scattered to the padded layout and go through SDPA as in base Soup, whose
mask gives padded keys probability exactly 0. Every other op works per position.

LoRA dropout draws random numbers per element, so it is applied to the scattered [B*T, hidden]
input, as in base Soup: the same number of draws in the same order gives the same mask at every
real position and leaves the RNG stream where base Soup leaves it. The LoRA matmuls then run on
the gathered rows. Anything this does not recognise (a LoRA variant, DoRA, merged or disabled
adapters, no autocast) runs the module on the scattered input and gathers the result back.

CHECKPOINT_LAYERS (off) recomputes every layer in backward instead: 2,277 MiB (6.3x less than
base Soup) for about 18% less speed. See the README.
"""

from __future__ import annotations

import functools

import torch

from . import unpad

ENABLED = True
# Recompute each packed layer in backward instead of keeping its activations: only the layer
# input ([n_real, hidden] fp32, ~6 KB per token) is kept per layer. Our own recomputation
# (torch.utils.checkpoint, non-reentrant, RNG preserved so LoRA dropout replays the same masks);
# Soup's gradient_checkpointing switch stays off. Costs a second forward of every layer.
CHECKPOINT_LAYERS = False  # True: 2,277 MiB but ~18% slower (README, Limits)
# Attention on the packed tokens themselves: PyTorch's memory-efficient attention kernel (the one
# SDPA already runs on the T4) called with cu_seqlens, causal within each row. No scatter to the
# padded batch, and no work on padded queries or keys. Falls back to padded SDPA (with a warning)
# if the op is unavailable.
VARLEN = True
_VARLEN_STATE = {"ok": True, "calls": 0}


class VarlenAttention(torch.autograd.Function):
    """Causal attention over B sequences packed back to back: q/k/v [n, heads, d] fp16."""

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_len, scale):
        out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            None,
            cu_seqlens,
            cu_seqlens,
            max_len,
            max_len,
            0.0,
            1,
            True,
            scale=scale,
        )  # 1 = causal from the top left
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens, seed, offset)
        ctx.max_len, ctx.scale = max_len, scale
        return out.squeeze(0)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse, cu_seqlens, seed, offset = ctx.saved_tensors
        gq, gk, gv, _ = torch.ops.aten._efficient_attention_backward(
            grad_out.unsqueeze(0).contiguous(),
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            None,
            out,
            cu_seqlens,
            cu_seqlens,
            ctx.max_len,
            ctx.max_len,
            lse,
            0.0,
            seed,
            offset,
            1,
            False,
            scale=ctx.scale,
        )
        return gq.squeeze(0), gk.squeeze(0), gv.squeeze(0), None, None, None


def _varlen_ok(attn, st):
    if (
        not (VARLEN and _VARLEN_STATE["ok"])
        or attn.config._attn_implementation != "sdpa"
    ):
        return None
    if attn.training and attn.attention_dropout:
        return None
    return st.varlen()


def _lora_parts(layer):
    """The vanilla LoRA adapters of a peft LoRA layer, or None if it is not one we handle."""
    if not all(
        hasattr(layer, a)
        for a in (
            "base_layer",
            "lora_A",
            "lora_B",
            "lora_dropout",
            "scaling",
            "active_adapters",
        )
    ):
        return None
    if getattr(layer, "merged", False) or getattr(layer, "disable_adapters", False):
        return None
    if not torch.is_autocast_enabled():
        return (
            None  # peft then casts the input to the LoRA dtype first; keep its own path
        )
    parts = []
    for name in layer.active_adapters:
        if name not in layer.lora_A.keys():
            continue
        if name in getattr(layer, "lora_variant", {}) or getattr(
            layer, "use_dora", {}
        ).get(name):
            return None
        parts.append(
            (
                layer.lora_A[name],
                layer.lora_B[name],
                layer.lora_dropout[name],
                layer.scaling[name],
            )
        )
    return parts


def _proj(layer, x, st):
    """layer(x) on packed rows, with LoRA dropout drawn over the padded batch as in base Soup."""
    if not hasattr(layer, "lora_A"):
        return layer(x)  # bitsandbytes Linear4bit (k_proj, o_proj): per row
    parts = _lora_parts(layer)
    if parts is None:
        return st.gather(layer(st.scatter(x)))
    result = layer.base_layer(x)
    for lora_a, lora_b, dropout, scaling in parts:
        if lora_a.weight.dtype != x.dtype:
            return st.gather(layer(st.scatter(x)))
        dropped = st.gather(
            dropout(st.scatter(x))
        )  # same draws as dropout on [B, T, hidden]
        result = result + lora_b(lora_a(dropped)) * scaling
    return result


def _attention(attn, x, attention_mask, position_embeddings, st, kwargs):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    n, d = x.shape[0], attn.head_dim
    q = _proj(attn.q_proj, x, st).view(n, -1, d)
    k = _proj(attn.k_proj, x, st).view(n, -1, d)
    v = _proj(attn.v_proj, x, st)
    cos, sin = st.rope(*position_embeddings)
    # Base: q [B, heads, T, d] with cos [B, 1, T, d]; here q [n, heads, d] with cos [n, 1, d].
    q, k = mq.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    varlen = _varlen_ok(attn, st)
    if varlen is not None:
        cu_seqlens, max_len = varlen
        heads, groups = q.shape[1], attn.num_key_value_groups

        def heads16(
            t,
        ):  # fp16 as SDPA's autocast makes it; kv heads repeated as repeat_kv does
            t = t.view(n, -1, d).to(torch.float16)
            if t.shape[1] != heads:
                t = (
                    t[:, :, None, :]
                    .expand(n, t.shape[1], groups, d)
                    .reshape(n, heads, d)
                )
            return t.contiguous()

        try:
            out = VarlenAttention.apply(
                heads16(q), heads16(k), heads16(v), cu_seqlens, max_len, attn.scaling
            )
            _VARLEN_STATE["calls"] += 1
            return attn.o_proj(out.reshape(n, -1))
        except Exception as exc:  # noqa: BLE001 - e.g. the op is missing in this torch build
            import warnings

            _VARLEN_STATE["ok"] = False
            warnings.warn(
                f"solution.packed: varlen attention unavailable, padded SDPA ({exc!r})",
                stacklevel=2,
            )

    def padded(t):  # [n, heads * d] -> [B, heads, T, d], zeros at the pads
        return st.scatter(t.reshape(n, -1)).view(st.b, st.t, -1, d).transpose(1, 2)

    interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, mq.eager_attention_forward
    )
    out, _ = interface(
        attn,
        padded(q),
        padded(k),
        padded(v),
        attention_mask,
        dropout=0.0 if not attn.training else attn.attention_dropout,
        scaling=attn.scaling,
        sliding_window=attn.sliding_window,
        **kwargs,
    )
    out = out.reshape(st.n_flat, -1).index_select(0, st.idx)  # [n, heads * d]
    return attn.o_proj(out)


def patch_qwen2_packed_layers():
    """Install after unpad.patch_qwen2_unpad() (it provides the token index and the final
    scatter); a layer that cannot run packed scatters its input and runs the original."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    cls = mq.Qwen2DecoderLayer
    if getattr(cls, "_soup_packed", False):
        return
    original = cls.forward

    @functools.wraps(original)
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        st = unpad.active()
        usable = (
            ENABLED
            and st is not None
            and st.pack_layers
            and past_key_values is None
            and position_embeddings is not None
            and self.self_attn.sliding_window is None
        )
        if hidden_states.dim() == 2 and st is not None and not usable:
            hidden_states = st.scatter(hidden_states).view(st.b, st.t, -1)
        if not usable:
            return original(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if hidden_states.dim() == 3:  # the first layer: embeddings [B, T, hidden]
            hidden_states = st.gather(hidden_states)
        attn_kwargs = dict(position_ids=position_ids, use_cache=use_cache, **kwargs)

        def body(
            h,
        ):  # everything it needs is captured here, so a recompute does not need unpad
            residual = h
            x = self.input_layernorm(h)
            h = residual + _attention(
                self.self_attn, x, attention_mask, position_embeddings, st, attn_kwargs
            )
            residual = h
            x = self.post_attention_layernorm(h)
            return residual + self.mlp(x)

        if CHECKPOINT_LAYERS and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(
                body, hidden_states, use_reentrant=False, preserve_rng_state=True
            )
        return body(hidden_states)

    cls.forward = forward
    cls._soup_packed = True
