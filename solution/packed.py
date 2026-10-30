"""Qwen2 decoder layers on the real tokens only; only attention itself sees the padded batch.

unpad.py already skips the pads in the MLP. Everything else in a decoder layer still ran on all
B*T positions: both RMSNorms, both residual additions, the q/k/v/o projections with their LoRA,
the fp32<->fp16 casts around them and the rotary embedding. In the base profile those
element-wise ops are about a quarter of the GPU time. Here the residual stream stays packed as
[n_real, hidden] from the first layer to the final norm (unpad.py scatters it back to
[B, T, hidden] when Qwen2Model returns), and a layer runs on the packed rows:

    input_layernorm -> q/k/v (+LoRA) -> rotary -> [scatter] SDPA [gather] -> o_proj -> + residual
    post_attention_layernorm -> MLP -> + residual

Only q/k/v are scattered to the padded [B, heads, T, d] layout for SDPA, whose mask hides the
pads: a padded key gets probability exactly 0, so its (now zero) value does not reach a real
query, and padded queries are not read. Every other op works per position, so the real rows get
the same values as before.

LoRA dropout draws random numbers per element, so it is applied to the scattered [B*T, hidden]
input, as in base Soup: the same number of draws in the same order gives the same mask at every
real position and leaves the RNG stream where base Soup leaves it. The LoRA matmuls then run on
the gathered rows. Anything this does not recognise (a LoRA variant, DoRA, merged or disabled
adapters, no autocast) runs the module on the scattered input and gathers the result back.
"""

from __future__ import annotations

import functools

import torch

from . import unpad

ENABLED = True


def _lora_parts(layer):
    """The vanilla LoRA adapters of a peft LoRA layer, or None if it is not one we handle."""
    if not all(hasattr(layer, a) for a in ("base_layer", "lora_A", "lora_B", "lora_dropout",
                                           "scaling", "active_adapters")):
        return None
    if getattr(layer, "merged", False) or getattr(layer, "disable_adapters", False):
        return None
    if not torch.is_autocast_enabled():
        return None  # peft then casts the input to the LoRA dtype first; keep its own path
    parts = []
    for name in layer.active_adapters:
        if name not in layer.lora_A.keys():
            continue
        if name in getattr(layer, "lora_variant", {}) or getattr(layer, "use_dora", {}).get(name):
            return None
        parts.append((layer.lora_A[name], layer.lora_B[name], layer.lora_dropout[name],
                      layer.scaling[name]))
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
        dropped = st.gather(dropout(st.scatter(x)))  # same draws as dropout on [B, T, hidden]
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

    def padded(t):  # [n, heads * d] -> [B, heads, T, d], zeros at the pads
        return st.scatter(t.reshape(n, -1)).view(st.b, st.t, -1, d).transpose(1, 2)

    interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation,
                                                         mq.eager_attention_forward)
    out, _ = interface(attn, padded(q), padded(k), padded(v), attention_mask,
                       dropout=0.0 if not attn.training else attn.attention_dropout,
                       scaling=attn.scaling, sliding_window=attn.sliding_window, **kwargs)
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
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        st = unpad.active()
        usable = (ENABLED and st is not None and st.pack_layers and past_key_values is None
                  and position_embeddings is not None and self.self_attn.sliding_window is None)
        if hidden_states.dim() == 2 and st is not None and not usable:
            hidden_states = st.scatter(hidden_states).view(st.b, st.t, -1)
        if not usable:
            return original(self, hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, position_embeddings=position_embeddings,
                            **kwargs)
        if hidden_states.dim() == 3:  # the first layer: embeddings [B, T, hidden]
            hidden_states = st.gather(hidden_states)

        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        attn_kwargs = dict(position_ids=position_ids, use_cache=use_cache, **kwargs)
        hidden_states = residual + _attention(self.self_attn, x, attention_mask,
                                              position_embeddings, st, attn_kwargs)
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(x)

    cls.forward = forward
    cls._soup_packed = True
