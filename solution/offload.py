"""Low GPU memory: frozen NF4 weights in pinned host RAM, streamed to the GPU one layer at a time.

What stays on the GPU in base Soup for the whole run, and what we do instead:

1. The packed NF4 bytes of all 196 Linear4bit layers (~655 MB). bitsandbytes quantizes each weight
   on the GPU while transformers loads the model; right after that we move the packed bytes to
   host RAM (`Params4bit._quantize` hook), so they never pile up on the GPU. When the model is
   ready they are copied into one pinned buffer per decoder layer. During training two GPU slots
   hold two layers: while layer i computes, layer i+1 (forward) or i-1 (backward) is copied in on
   a side stream (PCIe, ~23 MB per layer). CUDA events order the copy after the last kernel that
   used the slot, and the compute stream waits for the copy. The weights are the same bytes;
   our `nf4_dequant_kernel` dequantizes them exactly as before.

2. The activations of every layer. Each decoder layer is recomputed in backward (our own
   autograd Function; Soup's gradient_checkpointing switch stays off), so only the layer input
   [n_real, hidden] is kept, and that input is copied to pinned host RAM on a side stream and
   brought back for the layer's backward. The CUDA RNG state is saved per layer and restored for
   the recompute, so LoRA dropout draws the same masks, and the global RNG ends where base Soup
   leaves it. Autocast is re-entered with the forward's settings.

3. The tied fp16 embedding / head matrix (467 MB) is loaded straight into pinned host RAM. The
   embedding rows of a batch are gathered on the CPU and copied over; the fused head streams the
   vocabulary in 4096-row pieces through two GPU slots (head_stream.py).

4. transformers reserves the model's whole size on the GPU before loading (a load-speed trick,
   caching_allocator_warmup); we skip it, since most of that would now be host RAM.

5. peft's prepare_model_for_kbit_training widens the frozen tied embedding fp16 -> fp32
   (0.47 GB -> 0.93 GB, both alive during the cast). We leave it in fp16 and widen the embedding
   output to fp32 with a forward hook: the residual stream sees exactly the same fp32 numbers
   (an fp16 value widened is exact), and the head uses the fp16 weight as it already did
   (organizers allowed fp16 storage of the frozen tied embedding when exact).

The adapter (LoRA) and the optimizer state stay on the GPU. Nothing is reused between runs.
"""

from __future__ import annotations

import functools

import torch

from . import nf4

ENABLED = True  # stream NF4 weights + recompute layers (needs the store, see below)
EMBED_HOST = (
    True  # tied fp16 embedding / head weight in pinned host RAM (head_stream.py)
)
OFFLOAD_INPUTS = True  # keep the per-layer recompute inputs in pinned host RAM
STREAM_ABSMAX = (
    True  # the fp32 absmax travels with the bytes instead of an 82 MB GPU cache
)
QUANT_STATE_HOST = True  # bitsandbytes' 8-bit absmax (~20 MB) to host RAM once streamed
# In the recompute, stop each layer before down_proj: its forward matmul (and a dequant) only
# fed the layer output, whose gradient we already have; the gradient of down_proj's input is
# computed directly, the same matmul NF4Linear.backward does. Same numbers, ~29% fewer recompute
# GEMM flops. False: recompute the whole layer.
SKIP_DOWN_RECOMPUTE = True
_SPLIT_STATE = {"calls": 0}
ARENA_ROWS = 2048  # tokens per pinned input buffer (batch 4 x max_length 512)
_STORE = {"store": None}


def _align(n, to=16):
    return (n + to - 1) // to * to


def _cuda():
    return torch.device("cuda", torch.cuda.current_device())


def _is_nf4_linear(module) -> bool:
    w = getattr(module, "weight", None)
    return (
        type(w).__name__ == "Params4bit"
        and getattr(w, "quant_state", None) is not None
        and w.dtype == torch.uint8
    )


# ---------------------------------------------------------------- 1. weights in host RAM


def _apply_keep_host(self, fn, recurse=True):
    """Module._apply that leaves our host-RAM parameters where they are.

    accelerate / Trainer may call model.to(cuda) after Soup built the model; that would copy
    the streamed weights back to the GPU (and keep them there). Every other parameter, buffer
    and child is converted as usual."""
    hidden = {
        k: p for k, p in self._parameters.items() if getattr(p, "_soup_host", False)
    }
    for k in hidden:
        self._parameters[k] = None  # _apply skips None entries
    try:
        type(self)._apply(self, fn, recurse)
    finally:
        for k, p in hidden.items():
            self._parameters[k] = p
    return self


def _keep_on_host(module):
    module._apply = _apply_keep_host.__get__(module)


def patch_params4bit():
    """Packed NF4 bytes go to host RAM right after bitsandbytes quantized them on the GPU."""
    import bitsandbytes as bnb

    cls = bnb.nn.Params4bit
    if getattr(cls, "_soup_offload", False):
        return
    original = cls._quantize

    @functools.wraps(original)
    def _quantize(self, device):
        out = original(self, device)
        if ENABLED and self.data.is_cuda and self.data.dtype == torch.uint8:
            self.data = (
                self.data.cpu()
            )  # quant_state (absmax, ~20 MB in all) stays on the GPU
            self._soup_host = True
        return out

    cls._quantize = _quantize
    cls._soup_offload = True


class LayerStore:
    """Pinned host copies of each decoder layer's NF4 bytes and two GPU slots to stream them."""

    def __init__(self, layers):
        self.layers = list(layers)
        self.index = {id(layer): i for i, layer in enumerate(self.layers)}
        self.host, self.where = [], {}
        dev = _cuda()
        absmax_views = []  # (layer, offset, numel, quant_state): fp32 absmax inside the slot
        for i, layer in enumerate(self.layers):
            mods = [m for m in layer.modules() if _is_nf4_linear(m)]
            # the fp32 absmax nf4.py would cache on the GPU (82 MB for the model) travels with
            # the bytes instead: [packed bytes of every weight][fp32 absmax of every weight]
            absmax = (
                [nf4.absmax_fp32(m.weight.quant_state) for m in mods]
                if STREAM_ABSMAX
                else []
            )
            total = _align(sum(m.weight.numel() for m in mods))
            a_offs = []
            for a in absmax:
                a_offs.append(total)
                total = _align(total + a.numel() * 4)
            host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            off = 0
            for m in mods:
                w = m.weight
                n = w.numel()
                host[off : off + n].copy_(
                    w.data.reshape(-1)
                )  # from host or (tests) the GPU
                w.data = host[off : off + n].view(w.shape)
                w._soup_host = True
                _keep_on_host(m)
                self.where[w.data_ptr()] = (i, off, n)
                off += n
            for m, a, a_off in zip(mods, absmax, a_offs):
                host[a_off : a_off + a.numel() * 4].view(torch.float32).copy_(a)
                absmax_views.append((i, a_off, a.numel(), m.weight.quant_state))
            self.host.append(host)
        size = max((h.numel() for h in self.host), default=0)
        self.slots = [
            torch.empty(size, dtype=torch.uint8, device=dev) for _ in range(2)
        ]
        for i, a_off, numel, qs in absmax_views:
            # nf4.dequantize_nf4 reads this cache; the slot address of a weight never changes
            # (layer i always lands in slot i % 2 at the same offset), and its bytes and absmax
            # arrive in the same copy
            view = self.slots[i % 2][a_off : a_off + numel * 4].view(torch.float32)
            nf4._ABSMAX[id(qs)] = (qs, view)
            if QUANT_STATE_HOST:
                qs.to(
                    "cpu"
                )  # the 8-bit absmax and its state are no longer read on the GPU
        self.resident = [-1, -1]
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.free = [torch.cuda.Event(), torch.cuda.Event()]
        self.h2d = torch.cuda.Stream(device=dev)
        self.d2h = torch.cuda.Stream(device=dev)
        self.active = None  # the layer currently computing with its slot
        self.arena = {}  # layer -> pinned buffer for its recompute input (reused)

    # -- streaming
    def _issue(self, i):
        slot = i % 2
        self.h2d.wait_event(
            self.free[slot]
        )  # after the last kernel that read this slot
        with torch.cuda.stream(self.h2d):
            n = self.host[i].numel()
            self.slots[slot][:n].copy_(self.host[i], non_blocking=True)
            self.ready[slot].record(self.h2d)
        self.resident[slot] = i

    def ensure(self, i):
        if self.resident[i % 2] != i:
            self._issue(i)
        torch.cuda.current_stream().wait_event(self.ready[i % 2])
        self.active = i

    def prefetch(self, i):
        if 0 <= i < len(self.layers) and self.resident[i % 2] != i:
            self._issue(i)

    def release(self, i):
        self.free[i % 2].record(torch.cuda.current_stream())
        self.active = None

    def gpu_weight(self, B):
        """The GPU bytes of a host NF4 weight: the slot view while its layer runs here,
        otherwise a private copy (correct anywhere, e.g. outside training)."""
        hit = self.where.get(B.data_ptr())
        if hit is None:
            return None
        i, off, n = hit
        if self.active == i and self.resident[i % 2] == i:
            return self.slots[i % 2][off : off + n].view(B.shape)
        if STREAM_ABSMAX:
            # outside a streamed layer (no autograd use expected): the absmax lives in the
            # slot, so bring the whole layer in and read it from there
            self.ensure(i)
            return self.slots[i % 2][off : off + n].view(B.shape)
        return B.data.to(_cuda())

    # -- activations
    def to_host(self, h, i):
        """Layer i's recompute input into its own pinned buffer, allocated once and reused
        every step (pinned allocations of varying sizes stall the step)."""
        if not OFFLOAD_INPUTS:
            return (h.detach(), None)
        numel = h.numel()
        buf = self.arena.get(i)
        if buf is None or buf.numel() < numel or buf.dtype != h.dtype:
            # sized for the largest batch: batch * max_length tokens (4 x 512 here)
            cap = max(numel, ARENA_ROWS * h.shape[-1])
            buf = self.arena[i] = torch.empty(cap, dtype=h.dtype, pin_memory=True)
        host = buf[:numel].view(h.shape)
        # The previous step's copy back from this buffer was queued on the compute stream; the
        # side stream waits for everything queued there, so it is done before we overwrite it.
        self.d2h.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.d2h):
            host.copy_(h.detach(), non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.d2h)
        h.record_stream(self.d2h)
        return (host, done)

    def from_host(self, saved):
        t, done = saved
        if done is None:
            return t
        torch.cuda.current_stream().wait_event(done)
        return t.to(_cuda(), non_blocking=True)

    def check_once(self):
        """Report (once) if something moved our host-RAM weights back to the GPU."""
        if getattr(self, "_checked", False):
            return
        self._checked = True
        moved = [
            m
            for layer in self.layers
            for m in layer.modules()
            if _is_nf4_linear(m) and m.weight.is_cuda
        ]
        print(
            f"[solution.offload] first step: {len(moved)} NF4 weights on the GPU "
            f"(0 expected), {torch.cuda.memory_allocated() / 2**20:.0f} MiB allocated",
            flush=True,
        )

    def down_input_grad(self, down, act, grad_out):
        """d(residual + down(act))/d(act) seeded with grad_out, computed as autograd and
        NF4Linear.backward compute it: the fp32 grad rounded to act's fp16 by the add's
        backward, times the dequantized weight (one matmul, same shapes). None when down(act)
        would not take that path; the caller then runs down_proj as usual."""
        if hasattr(down, "lora_A") or getattr(down, "bias", None) is not None:
            return None
        if down._forward_hooks or down._forward_pre_hooks or down._backward_hooks:
            return None
        w = getattr(down, "weight", None)
        qs = getattr(w, "quant_state", None)
        if qs is None or type(w).__name__ != "Params4bit":
            return None
        gpu = w if w.is_cuda else self.gpu_weight(w)
        if gpu is None or not nf4._usable(act, gpu, qs, None, None):
            return None
        if grad_out.shape != (act.shape[0], qs.shape[0]) or grad_out.dtype != torch.float32:
            return None
        g16 = grad_out.to(act.dtype)
        return torch.matmul(g16, nf4.dequantize_nf4(gpu, qs, g16.dtype))

    def run_layer(self, layer, body, h):
        self.check_once()
        params = tuple(p for p in layer.parameters() if p.requires_grad)
        try:
            ac = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        except TypeError:  # older torch
            ac = (torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype())
        return _StreamedLayer.apply(h, (self, self.index[id(layer)], body, ac), *params)


class _StreamedLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, info, *params):
        store, i, body, ac = info
        ctx.info, ctx.n_params = info, len(params)
        ctx.rng = torch.cuda.get_rng_state()
        store.ensure(i)
        store.prefetch(i + 1)
        with torch.no_grad():
            out = body(h)
        store.release(i)
        ctx.saved = store.to_host(h, i)
        ctx.params = params
        return out

    @staticmethod
    def backward(ctx, grad_out):
        store, i, body, (ac_on, ac_dtype) = ctx.info
        need_h = ctx.needs_input_grad[0]
        h = store.from_host(ctx.saved).detach().requires_grad_(need_h)
        ctx.saved = None
        store.ensure(i)
        store.prefetch(i - 1)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.cuda.set_rng_state(ctx.rng)
            with (
                torch.enable_grad(),
                torch.autocast("cuda", dtype=ac_dtype, enabled=ac_on),
            ):
                out = body(h, split=SKIP_DOWN_RECOMPUTE)
                g_act = None
                if isinstance(out, tuple):  # stopped before down_proj (packed.py)
                    residual, act, down = out
                    if residual.requires_grad and act.requires_grad:
                        g_act = store.down_input_grad(down, act, grad_out)
                    if g_act is None:
                        out = residual + down(act)  # what the full body returns
        inputs = ((h,) if need_h else ()) + ctx.params
        if g_act is not None:
            # out = residual + down(act), and act feeds nothing else: seed both directly.
            _SPLIT_STATE["calls"] += 1
            grads = torch.autograd.grad(
                (residual, act), inputs, (grad_out, g_act), allow_unused=True
            )
        else:
            grads = torch.autograd.grad(out, inputs, grad_out, allow_unused=True)
        store.release(i)
        if need_h:
            return (grads[0], None) + tuple(grads[1:])
        return (None, None) + tuple(grads)


def active_store():
    return _STORE["store"] if ENABLED else None


def build_store(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, torch.nn.ModuleList) or not torch.cuda.is_available():
        return None
    if not any(_is_nf4_linear(m) for m in layers.modules()):
        return None
    store = LayerStore(layers)
    _STORE["store"] = store
    torch.cuda.empty_cache()
    print(
        f"[solution.offload] NF4 weights of {len(store.layers)} layers in pinned host RAM "
        f"({sum(h.numel() for h in store.host) / 2**20:.0f} MiB)",
        flush=True,
    )
    return store


# ---------------------------------------------------------------- bitsandbytes / peft hooks


def patch_matmul_4bit():
    """Outermost wrapper of bnb.matmul_4bit (install after nf4.patch_bnb_matmul_4bit): a weight
    in host RAM is handed to the inner function as its GPU bytes."""
    import bitsandbytes as bnb

    if getattr(bnb, "_soup_offload", False):
        return
    inner = bnb.matmul_4bit

    @functools.wraps(inner)
    def matmul_4bit(A, B, quant_state=None, out=None, bias=None):
        if isinstance(B, torch.Tensor) and not B.is_cuda and A.is_cuda:
            store = _STORE["store"]
            gpu = store.gpu_weight(B) if store is not None else None
            B = gpu if gpu is not None else B.data.to(A.device)
        return inner(A, B, quant_state=quant_state, out=out, bias=bias)

    bnb.matmul_4bit = matmul_4bit
    bnb._soup_offload = True


def _upcast_output(module, inputs, output):
    return output.float()


def _host_embedding_forward(self, input_ids):
    """nn.Embedding.forward with the fp16 table in pinned host RAM: the rows are gathered on the
    CPU (a few MB per step) and copied to the GPU. Same values as the lookup on the GPU."""
    w = self.weight
    if w.is_cuda:
        return torch.nn.functional.embedding(
            input_ids,
            w,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
    ids = input_ids.to("cpu")
    rows = torch.empty((*ids.shape, w.shape[1]), dtype=w.dtype, pin_memory=True)
    torch.index_select(w.detach(), 0, ids.reshape(-1), out=rows.view(-1, w.shape[1]))
    return rows.to(input_ids.device, non_blocking=True)


def _host_head_forward(self, x):
    """lm_head outside the fused training head (not used in training): weight copied over."""
    w = self.weight
    if w.is_cuda or not x.is_cuda:
        return torch.nn.functional.linear(x, w, self.bias)
    return torch.nn.functional.linear(x, w.to(x.device, non_blocking=True), self.bias)


def move_embedding_to_host(model):
    """The tied fp16 embedding / head matrix (467 MB) to pinned host RAM."""
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    w = embed.weight
    if (
        not EMBED_HOST
        or w.dtype != torch.float16
        or w.requires_grad
        or embed.max_norm is not None
        or (head is not None and head.weight is not w)
    ):
        if (
            not w.is_cuda
        ):  # loaded to host RAM (patch_loader_device) but not usable there
            w.data = w.data.to(_cuda())
        return False
    if w.is_cuda:
        host = torch.empty(w.shape, dtype=w.dtype, pin_memory=True)
        host.copy_(w.data)
    else:
        host = w.data if w.data.is_pinned() else w.data.pin_memory()
    w.data = host
    w._soup_host = True
    _keep_on_host(embed)
    embed.forward = _host_embedding_forward.__get__(embed)
    if head is not None:
        _keep_on_host(head)
        head.forward = _host_head_forward.__get__(head)
    torch.cuda.empty_cache()
    print(
        "[solution.offload] tied fp16 embedding / head in pinned host RAM", flush=True
    )
    return True


def patch_model_device():
    """PreTrainedModel.device is its first parameter's device, and the first parameter is now the
    embedding in host RAM. The model computes on the GPU, so skip the parameters we keep in host
    RAM (inputs are still moved by the Trainer to args.device either way)."""
    from transformers import modeling_utils

    cls = modeling_utils.PreTrainedModel
    original = cls.device
    if getattr(original.fget, "_soup_offload", False):
        return

    def device(self):
        for p in self.parameters():
            if not getattr(p, "_soup_host", False):
                return p.device
        return original.fget(self)

    prop = property(device)
    prop.fget._soup_offload = True
    cls.device = prop


def patch_peft():
    import peft

    if getattr(peft, "_soup_offload", False):
        return
    prepare = peft.prepare_model_for_kbit_training
    get_peft_model = peft.get_peft_model

    @functools.wraps(prepare)
    def prepare_model_for_kbit_training(model, *args, **kwargs):
        if not ENABLED:
            return prepare(model, *args, **kwargs)
        embed = model.get_input_embeddings()
        head = model.get_output_embeddings()
        w = getattr(embed, "weight", None)
        keep16 = (
            isinstance(embed, torch.nn.Embedding)
            and w is not None
            and w.dtype == torch.float16
            and (head is None or head.weight is w)
        )
        if keep16:  # hide it from the fp16 -> fp32 cast, then put it back as it was
            data = w.data
            w.data = torch.empty(0, dtype=torch.float32, device=data.device)
        try:
            model = prepare(model, *args, **kwargs)
        finally:
            if keep16:
                w.data = data
        if keep16:
            embed.register_forward_hook(
                _upcast_output
            )  # the fp32 output the cast would give
        build_store(model)
        if keep16 and _STORE["store"] is not None:
            move_embedding_to_host(model)
        else:
            w = getattr(model.get_input_embeddings(), "weight", None)
            if (
                w is not None and not w.is_cuda
            ):  # loaded to host RAM: back as base Soup has it
                w.data = w.data.to(_cuda())
        return model

    @functools.wraps(get_peft_model)
    def get_peft_model_(model, *args, **kwargs):
        model = get_peft_model(model, *args, **kwargs)
        if _STORE["store"] is not None:
            # peft puts each new adapter on its base layer's device: host RAM for our NF4
            # weights. The adapter trains on the GPU, so move it there before the optimizer.
            dev = _cuda()
            for p in model.parameters():
                if p.requires_grad and not p.is_cuda:
                    p.data = p.data.to(dev)
        return model

    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    peft.get_peft_model = get_peft_model_
    peft._soup_offload = True


def patch_loader_warmup():
    """transformers reserves the whole model's size on the GPU in one block before loading
    (caching_allocator_warmup, a load-speed trick). With the weights going to host RAM that block
    would be the peak of the run. Skipping it changes no value; tensors are allocated as loaded."""
    from transformers import modeling_utils

    if getattr(modeling_utils, "_soup_offload", False):
        return
    original = modeling_utils.caching_allocator_warmup

    @functools.wraps(original)
    def caching_allocator_warmup(*args, **kwargs):
        if ENABLED:
            return None
        return original(*args, **kwargs)

    modeling_utils.caching_allocator_warmup = caching_allocator_warmup
    modeling_utils._soup_offload = True


def patch_loader_device():
    """Load the tied embedding (467 MB in fp16) straight into host RAM instead of loading it on
    the GPU and moving it afterwards: the device transformers picks for that one tensor."""
    from transformers import core_model_loading as cml

    if getattr(cml, "_soup_offload", False):
        return
    original = cml.get_device

    @functools.wraps(original)
    def get_device(device_map, name, *args, **kwargs):
        if ENABLED and EMBED_HOST and str(name).endswith("embed_tokens.weight"):
            return "cpu"
        return original(device_map, name, *args, **kwargs)

    cml.get_device = get_device
    cml._soup_offload = True


def patch_offload():
    if not ENABLED:
        return
    patch_loader_warmup()
    patch_loader_device()
    patch_model_device()
    patch_params4bit()
    patch_matmul_4bit()
    patch_peft()
    if EMBED_HOST:
        from .head_stream import start_warmup

        start_warmup()  # compiles the streamed head kernels while Soup loads the model
