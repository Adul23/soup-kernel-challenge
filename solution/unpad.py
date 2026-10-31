"""Unpadding for the Qwen2 MLP: run gate/up/down only on real (non-pad) tokens.

TRL right-pads every batch to its longest row. The MLP is position-wise, so the output at a pad
position never reaches a real token (attention masks pad keys, the loss masks pad labels, TRL's
entropy is weighted by attention_mask). We gather the real rows, run the MLP on them, and scatter
the result back into a zero tensor.

Where the mask comes from: Qwen2MLP.forward only gets hidden states, and pad hidden states are not
zero vectors (the pad token has a real embedding, and attention + residuals make every row dense
from layer 0 on), so the mask cannot be recovered from `x`. Comparing input_ids to pad_token_id is
also unsafe (the pad id can be a real token in the text). The only ground truth is the 2-D
attention_mask the collator builds, so we read it where it enters the model, in Qwen2Model.forward,
turn it into a flat index once per forward pass (one host sync per step instead of one per layer),
and hand that index to the 28 MLP calls through a module-level slot.

Autograd: index_select backward is index_add into zeros (pad rows get exactly zero gradient, as in
the base run, where nothing downstream depends on them); index_copy backward is index_select of the
incoming gradient. No custom backward, no clone, no detach. Dtypes are untouched: the packed rows
go through the same forward as before (in this setup the MLP input is fp32 because the norm weights
are fp32, and bitsandbytes casts to fp16 for the matmul and back), and zeros are exact in any dtype.
"""

import functools

import torch

ENABLED = True  # tests flip this to get the reference path
PACK_LAYERS = (
    True  # also keep the residual stream packed between decoder layers (packed.py)
)


class State:
    """The real tokens of the Qwen2Model forward that is running."""

    def __init__(self, idx, b, t, pack, mask=None):
        self.idx, self.b, self.t = idx, b, t
        self.n_flat = b * t
        self.pack_layers = pack  # packed.py may keep hidden states as [n_real, hidden]
        self.mask = mask  # the 2-D attention_mask this index was taken from
        self._rope = {}
        self._varlen = None

    def varlen(self):
        """(cu_seqlens int32 [B+1], longest row) when every row's real tokens are a prefix of
        the row (right padding), so the packed rows are B whole sequences back to back; else
        None. Computed once per forward (one host sync)."""
        if self._varlen is None:
            self._varlen = False
            if isinstance(self.mask, torch.Tensor):
                real = self.mask != 0
                lengths = real.sum(1, dtype=torch.int32)
                prefix = (
                    torch.arange(self.t, device=real.device)[None, :] < lengths[:, None]
                )
                if bool(torch.equal(prefix, real)) and int(lengths.min()) > 0:
                    cu = torch.nn.functional.pad(
                        lengths.cumsum(0, dtype=torch.int32), (1, 0)
                    )
                    self._varlen = (cu, int(lengths.max()))
        return self._varlen or None

    def gather(self, x):
        """[B, T, d] (or [B*T, d]) -> [n_real, d]."""
        return x.reshape(self.n_flat, x.shape[-1]).index_select(0, self.idx)

    def scatter(self, x):
        """[n_real, d] -> [B*T, d] with zeros at the pads (autograd-safe, out of place)."""
        return x.new_zeros((self.n_flat, x.shape[-1])).index_copy(0, self.idx, x)

    def rope(self, cos, sin):
        """cos/sin [B or 1, T, d] gathered to [n_real, d], once per forward."""
        key = (id(cos), id(sin))
        hit = self._rope.get(key)
        if hit is None:
            hit = self._rope[key] = (
                cos.expand(self.b, self.t, cos.shape[-1])
                .reshape(self.n_flat, -1)
                .index_select(0, self.idx),
                sin.expand(self.b, self.t, sin.shape[-1])
                .reshape(self.n_flat, -1)
                .index_select(0, self.idx),
                cos,
                sin,
            )  # keep the originals alive so their ids stay theirs
        return hit[0], hit[1]


_ACTIVE: State | None = None


def active():
    return _ACTIVE


def token_index(attention_mask, batch, seq_len):
    """Flat int64 index of the real tokens, or None when unpadding does not apply."""
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 2:
        return None  # 4-D masks, dicts of masks (generate) or no mask: leave alone
    if tuple(attention_mask.shape) != (batch, seq_len):
        return None  # KV-cache decoding: mask covers past tokens too
    flat = attention_mask.reshape(-1) != 0
    idx = flat.nonzero().squeeze(1)  # the one host sync of the step
    if idx.numel() == 0 or idx.numel() == flat.numel():
        return None  # nothing to skip
    return idx


def unpadded_call(fn, x, idx):
    """fn(x) for a position-wise fn, computed only on the rows `idx` of x.view(-1, hidden)."""
    b, t, h = x.shape
    x_packed = x.reshape(b * t, h).index_select(0, idx)  # [n_real, h]
    y_packed = fn(x_packed.unsqueeze(0)).squeeze(0)  # [n_real, h_out], same dtype path
    y = y_packed.new_zeros((b * t, y_packed.shape[-1]))
    y = y.index_copy(0, idx, y_packed)  # out of place: autograd-safe
    return y.view(b, t, -1)


def patch_qwen2_unpad():
    """Install the unpadding. Call it after any other Qwen2MLP.forward patch: it wraps whatever
    forward is installed at that moment (the example SwiGLU or your own kernel)."""
    from transformers.models.qwen2 import modeling_qwen2

    model_cls, mlp_cls = modeling_qwen2.Qwen2Model, modeling_qwen2.Qwen2MLP
    if getattr(mlp_cls, "_soup_unpad", False):
        return

    model_forward = model_cls.forward
    mlp_forward = mlp_cls.forward

    @functools.wraps(model_forward)
    def forward_model(self, *args, **kwargs):
        global _ACTIVE
        state = None
        if ENABLED and self.training and kwargs.get("past_key_values") is None:
            mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            ref = kwargs.get("input_ids", args[0] if args else None)
            if ref is None:
                ref = kwargs.get("inputs_embeds")
            if isinstance(ref, torch.Tensor) and ref.dim() >= 2:
                b, t = ref.shape[0], ref.shape[1]
                idx = token_index(mask, b, t)
                if idx is not None:
                    pack = (
                        PACK_LAYERS
                        and kwargs.get("return_dict") is not False
                        and not kwargs.get("output_hidden_states")
                        and not kwargs.get("output_attentions")
                    )
                    state = State(idx.to(ref.device), b, t, pack, mask=mask)
        previous, _ACTIVE = _ACTIVE, state
        try:
            out = model_forward(self, *args, **kwargs)
            hidden = getattr(out, "last_hidden_state", None)
            if (
                state is not None
                and isinstance(hidden, torch.Tensor)
                and hidden.dim() == 2
            ):
                # packed.py kept the stream packed through the layers and the final norm:
                # hand the usual [B, T, hidden] back, zeros at the pads.
                out["last_hidden_state"] = state.scatter(hidden).view(
                    state.b, state.t, -1
                )
            return out
        finally:
            _ACTIVE = previous

    @functools.wraps(mlp_forward)
    def forward_mlp(self, x):
        state = _ACTIVE
        if state is None or x.dim() != 3 or x.shape[0] * x.shape[1] != state.n_flat:
            return mlp_forward(
                self, x
            )  # also the packed [n_real, hidden] input of packed.py
        return unpadded_call(lambda inp: mlp_forward(self, inp), x, state.idx)

    model_cls.forward = forward_model
    mlp_cls.forward = forward_mlp
    mlp_cls._soup_unpad = True
