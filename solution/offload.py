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

3. transformers reserves the model's whole size on the GPU before loading (a load-speed trick,
   caching_allocator_warmup); we skip it, since most of that would now be host RAM.

4. peft's prepare_model_for_kbit_training widens the frozen tied embedding fp16 -> fp32
   (0.47 GB -> 0.93 GB, both alive during the cast). We leave it in fp16 and widen the embedding
   output to fp32 with a forward hook: the residual stream sees exactly the same fp32 numbers
   (an fp16 value widened is exact), and the head uses the fp16 weight as it already did
   (organizers allowed fp16 storage of the frozen tied embedding when exact).

The adapter (LoRA) and the optimizer state stay on the GPU. Nothing is reused between runs.
"""

from __future__ import annotations

import functools

import torch

ENABLED = True  # stream NF4 weights + recompute layers (needs the store, see below)
OFFLOAD_INPUTS = True  # keep the per-layer recompute inputs in pinned host RAM
_STORE = {"store": None}


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
        for i, layer in enumerate(self.layers):
            mods = [m for m in layer.modules() if _is_nf4_linear(m)]
            total = sum(m.weight.numel() for m in mods)
            host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            off = 0
            for m in mods:
                w = m.weight
                n = w.numel()
                host[off : off + n].copy_(
                    w.data.reshape(-1)
                )  # from host or (tests) the GPU
                w.data = host[off : off + n].view(w.shape)
                self.where[w.data_ptr()] = (i, off, n)
                off += n
            self.host.append(host)
        size = max((h.numel() for h in self.host), default=0)
        self.slots = [
            torch.empty(size, dtype=torch.uint8, device=dev) for _ in range(2)
        ]
        self.resident = [-1, -1]
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.free = [torch.cuda.Event(), torch.cuda.Event()]
        self.h2d = torch.cuda.Stream(device=dev)
        self.d2h = torch.cuda.Stream(device=dev)
        self.active = None  # the layer currently computing with its slot

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
        return B.data.to(_cuda())

    # -- activations
    def to_host(self, h):
        if not OFFLOAD_INPUTS:
            return (h.detach(), None)
        host = torch.empty(h.shape, dtype=h.dtype, pin_memory=True)
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

    def run_layer(self, layer, body, h):
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
        ctx.saved = store.to_host(h)
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
                out = body(h)
        inputs = ((h,) if need_h else ()) + ctx.params
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


def patch_offload():
    if not ENABLED:
        return
    patch_loader_warmup()
    patch_params4bit()
    patch_matmul_4bit()
    patch_peft()
