# Soup Kernel Challenge: 21x less GPU memory (676 MiB), 1.24x faster on a T4

Our solution runs `soup train` (Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4 + LoRA, 64 steps on a
Kaggle T4) with a peak GPU memory of **676 MiB instead of 14,468 MiB (21.4x less)** and
**1.24x faster** (71.6 s instead of 88.9 s). Eval loss stays the same (−0.001%), and all 10
checks of `harness/check_rules.py` pass. Six `@triton.jit` kernels of our own take 9.2% of the
training time. Everything lives in `solution/` and is installed by `solution.apply()`.

How: the frozen model lives in pinned host RAM. The NF4 weights stream to the GPU one decoder
layer at a time over PCIe while the previous layer computes; every layer is recomputed in
backward, and its input waits in host RAM; the 151,936 x 1,536 tied embedding / head matrix
streams in 4,096-row pieces through our fused head. The GPU holds the CUDA context, the LoRA
adapter and its optimizer state, two layer slots, two head slots and one layer's activations.

**Public Kaggle notebook (the measured run):** https://www.kaggle.com/code/djparty/soupv4  
**Final commit:** https://github.com/BAITC-Hacks/hack-1d8bdb00-justdoner/commit/<FINAL_SHA>

| Kaggle T4, `harness/run_task.py` | Base Soup | Ours | Ratio |
|---|---|---|---|
| Wall time of `soup train`, s | 88.93 | 71.64 | **1.241x faster** |
| Training loop (64 steps), s | — | 51.54 | |
| Peak GPU memory (nvidia-smi), MiB | 14,468 | 676 | **21.4x less** |
| Peak memory allocated by PyTorch, MiB | — | 392 | |
| Eval loss, public rows | 2.04968 | 2.04966 | −0.001% |
| Loss curve vs base (check 6) | | first step 0.02%, mean 0.01% | limits 1% / 3% |

The nvidia-smi peak is the sum over all GPUs and includes any CUDA context another process
(e.g. the notebook kernel itself) holds when the run starts; in that run base Soup peaked at
14,468 MiB instead of its usual 14,345, so about 120 MiB of both peaks were not ours. Base
Soup's wall time also varies between Kaggle sessions (75–92 s).

We also keep the faster, larger variant in the history: without the offload (section 8) the same
code runs in 45.9 s (1.63x against a 75 s base) at 4,655 MiB (3.08x).

## What we did

We followed one rule throughout: **change how things are computed, never what is computed.**
Most patches reproduce base Soup's arithmetic bit for bit, and our tests check that with
`torch.equal`. The others reorder only fp32 or fp16 accumulation, and their tests compare them to
the dense path within tight tolerances. All patches are class-level and narrow. Each one checks
that it is in the exact situation it was written for: training, fp16 autocast, bitsandbytes NF4
layers, Qwen2 shapes. Anywhere else it calls the original code, so Soup's 79 regression tests
pass unchanged.

### 1. Fused output head + loss: no `[B, T, 151,936]` logits (`head.py`, kernel `fused_head_ce_kernel`)

In base Soup the 151,936-word head builds fp16 logits. accelerate widens them to fp32 for TRL,
`ForCausalLMLoss` makes another fp32 copy, and backward keeps fp32 and fp16 gradients of the
same size. TRL reads the logits again for entropy and token accuracy, and Soup's rewind
recorder reads them a third time for per-row losses. At the peak that is several GB.

Our version:
- **Chunked logits.** cuBLAS computes the logits for 128 tokens at a time into one reused fp16
  buffer (with the offload, the vocabulary is chunked instead: section 8). The fp16 inputs, fp32 accumulation and fp16 output are what autocast does.
- **One kernel pass per row.** Our kernel reads each row once and computes in fp32 an online
  max / sum-exp (log-sum-exp), the entropy, the first-index argmax and the target logit.
- **Gradient in place.** For labelled rows, a second pass writes `softmax - onehot` back into
  the buffer, scaled by 256 so that small probabilities do not underflow in fp16.
- **Skipped work.** Pad positions are skipped entirely, and prompt positions get no gradient
  pass.
- **Soup outputs kept.** TRL's `entropy`, `num_tokens` and `mean_token_accuracy` are logged with
  TRL's own formulas, from the per-token results. Soup's rewind recorder gets per-token losses
  through Soup's own no-logits contract (`trainer._soup_chunked_step`, Soup #1693). We did not
  turn on Soup's `stream_loss_chunk` switch, and we did not reuse its code.
- **One fp16 copy of the tied weight.** prepare_model_for_kbit_training widens the tied embedding
  / head weight to fp32 (0.93 GB). We check, in row chunks, that every value is an fp16 value
  (the model is loaded in fp16), and then store it once in fp16. The head uses it directly, and
  a forward hook widens the embedding output back to fp32. The stream keeps its dtype and every
  value. If any value is not fp16-exact, the weight is left as it is. With the offload (section 8)
  the weight is never widened in the first place.

### 2. NF4 weight dequantization (`nf4.py`, kernel `nf4_dequant_kernel`)

Every bitsandbytes `Linear4bit` dequantizes its NF4 weight to fp16 twice per step: in forward
(gemm_4bit falls back to dequantize + `F.linear` for these shapes) and in backward. That is 392
times per step. Each time it also first rebuilds the fp32 absmax from its 8-bit double
quantization.

Our version:
- **Our kernel writes the fp16 weight straight from the packed bytes.** It uses the arithmetic
  of bitsandbytes' `csrc/kernels.cu`: `LUT[byte >> 4] * absmax` and `LUT[byte & 15] * absmax`,
  one fp32 multiply each, rounded to fp16 on store.
- **The absmax of each frozen weight is rebuilt once**, by bitsandbytes itself, and cached
  (about 80 MB).
- **Launch arguments are cached per weight** (flat packed view, absmax, LUT, size, grid): the
  hook runs 389 times a step, so the per-call Python work is a dict lookup plus a storage check
  (`data_ptr`, size, dtype, device), since Linear4bit passes a fresh `.t()` view each call. The
  launch itself still goes through `nf4_dequant_kernel[grid](...)`.
- **Same numbers:** the output is bit-identical to bitsandbytes (`test_nf4.py`). The matmuls
  stay in cuBLAS, as in the base.

### 3. RMSNorm (`rmsnorm.py`, kernels `_rmsnorm_forward`, `_rmsnorm_backward_input`)

These are Triton forward and input-gradient kernels for `Qwen2RMSNorm`. They keep the base's
rounding order: normalize in fp32, round to the input dtype, multiply by the fp32 weight.

### 4. MLP: pads skipped, activations in fp16 (`unpad.py`, `mlp_fp16.py`)

- **Pads skipped.** The MLP runs only on real tokens: the rows are gathered by an index taken
  once per forward from the 2-D `attention_mask`, then scattered back. Pad hidden states are
  not zero vectors, so the mask must come from `attention_mask`.
- **Activations in fp16.** The MLP input is fp32, because the norm weights are fp32. So each
  `Linear4bit` cast it to fp16, multiplied in fp16 and widened the result back to fp32, which
  kept `gate` and `up` as fp32 `[tokens, 8960]` tensors holding only fp16 values. We cast the
  input once per projection, and gate, up, SwiGLU and down stay fp16.
- **Same numbers:** this is bit-identical to the previous path (`test_mlp_fp16.py`). Two
  separate casts keep the input gradient summed in fp32, exactly as through bitsandbytes' two
  internal casts.
- **SwiGLU kernel:** the SwiGLU itself is still the starter kit's example kernel. It does not
  count toward check 4, and we do not claim it.

### 5. Decoder layers on real tokens only, attention included (`packed.py`)

The residual stream stays packed as `[n_real, 1536]` from the first layer to the final norm.
Both RMSNorms, both residual additions, the q/k/v/o projections with their LoRA, the casts
around them, the rotary embedding and attention itself run only on real tokens.

Attention uses PyTorch's memory-efficient attention kernel, the one SDPA already runs on the T4,
called on the packed tokens with `cu_seqlens` and a causal mask within each row. It never
touches a padded query or key, and q/k/v are not copied into a padded batch. This needs right
padding (each row's real tokens a prefix), which is what TRL produces. Otherwise, or if the op is
missing, q/k/v are scattered into the padded layout for SDPA, as in base Soup.

LoRA dropout draws one random number per element, so we apply it to the scattered `[B*T, 1536]`
input, as base Soup does: the same number of draws in the same order. The masks at real
positions are then the same, and the RNG stream ends where base Soup leaves it. `test_packed.py`
checks this by comparing CUDA RNG states. The pad rows of that scattered input are left
uninitialised (`State.scatter_for_dropout`): dropout is per element, and the pad rows are
discarded by the gather right after it and get no gradient, so only the shape matters
(`test_unpad.py` checks equal outputs, gradients and RNG state).

### 6. Kernel compilation off the critical path (`warmup.py`)

Every measured run starts with empty Triton caches. Compiling our kernels on the first launches
added about 4 s to step 1. A thread inside the same `soup train` process compiles them
(`JITFunction.warmup`) while Soup loads and quantizes the model, and the trainer waits for it
before step 1. It is the same compilation, timed inside the same command, only earlier. Nothing
is cached across runs. Setting `ENABLED = False` in `warmup.py` restores compile-on-first-use.
We asked the organizers before relying on this (a thread in the measured process, a fresh cache,
no other process), and on the fp16 storage of the frozen tied embedding; both were allowed.

### 7. CPU overhead of the `soup train` command itself (`startup.py`)

Two pieces of startup work do not affect the training. We asked the organizers and they allowed
both, under the conditions noted below. They run inside the measured process, keep nothing
between runs, and behave the same under measurement and tests. They do not count toward check 4.

- **`importlib.metadata.packages_distributions()`** walks every installed package, about 2.3 s
  on the Kaggle image. transformers computes it at import, before `apply()` runs, and keeps it;
  trl computes it again when it is imported. Later callers now get the mapping transformers
  already computed in this process with the same function. Every caller gets a fresh copy, so
  no caller can change what another one sees.
- **`datasets.Dataset.map`** hashes the transform with dill only to name a cache file. Soup
  builds its datasets in memory (`Dataset.from_list`), where no cache file is ever written or
  read, and the transform closes over the tokenizer, which is slow to hash. For such datasets
  (no cache files, no `cache_file_name`, one process) we pass a random `new_fingerprint`, as
  `datasets` itself does when hashing fails. It comes from `datasets`' own `random.Random`, so
  the global, NumPy and torch random states are untouched. The mapped rows, their order and
  everything downstream are identical; Soup's rewind fingerprint hashes the rows themselves.
  `test_startup.py` checks the rows, the random states and that no hashing happened.

### 8. Low GPU memory: the frozen model in host RAM (`offload.py`, `head_stream.py`)

Sections 1–7 make each step cheaper; this section moves what does not have to be on the GPU off
it. It is our own code (Soup's `gradient_checkpointing` and layer-streaming switches stay off), and
it changes no value: the same bytes are dequantized by the same kernel, and the same fp16 logits
go into the same online statistics.

- **NF4 weights stream layer by layer (625 MiB in host RAM).** Right after bitsandbytes quantizes
  a weight on the GPU during loading, its packed bytes move to host RAM (`Params4bit._quantize`
  hook), so they never pile up on the GPU. Once the model is ready they are copied into one
  pinned buffer per decoder layer. Two GPU slots hold two layers: while layer i computes, layer
  i+1 (forward) or i−1 (backward) is copied in on a side CUDA stream (~23 MB per layer; PCIe
  measured at 11.7 GB/s on the Kaggle T4, against 2–4 ms of compute per layer). CUDA events order
  each copy after the last kernel that read the slot, and the compute stream waits for the copy.
  The quantization state (absmax) moves to host RAM too: the fp32 absmax our kernel reads (82 MB as a GPU cache) is copied with each layer's bytes into its slot, so bitsandbytes' 8-bit absmax is no longer read on the GPU.
- **Every layer is recomputed in backward, and its input waits in host RAM.** Our own autograd
  Function (`_StreamedLayer`) runs the packed layer without autograd in forward, copies its input
  `[n_real, 1536]` to a pinned buffer on another side stream, and in backward copies it back,
  recomputes the layer with autograd and backpropagates through it. The CUDA RNG state is saved
  per layer and restored for the recompute, so LoRA dropout draws the same masks, and the global
  RNG ends where base Soup leaves it (`test_offload.py` compares RNG states, loss and gradients
  against the plain run). Autocast is re-entered with the forward's settings.
- **The tied embedding / head matrix (467 MB in fp16) lives in pinned host RAM.** It is loaded
  there directly (the device transformers picks for `embed_tokens.weight`), and kept in fp16:
  `prepare_model_for_kbit_training` would widen it to fp32, so we hide it from that cast and widen
  the embedding output instead (an fp16 value widened is exact, so the residual stream sees the
  same fp32 numbers; the organizers allowed fp16 storage of the frozen tied embedding). The
  embedding rows of a batch are gathered on the CPU and copied over.
- **The fused head streams the vocabulary (`head_stream.py`).** The vocabulary becomes the outer
  loop: 4,096-row pieces of the weight come to the GPU through two slots (the next piece copied
  while the current one is used), and each piece is used for all token rows at once. Pass 1:
  cuBLAS logits of the piece, then `head_chunk_stats_kernel` carries each row's online max /
  sum-exp / entropy sum / first-index argmax / target logit across pieces in fp32. The piece size
  is a multiple of the 4,096-wide block, so the blocks are visited in the same order with the same
  arithmetic as `fused_head_ce_kernel`. Pass 2 (labelled rows): logits again,
  `head_chunk_grad_kernel` turns them in place into `256 * (softmax − onehot)`, and cuBLAS
  multiplies them by the piece; the fp16 products are summed in fp32. `test_head_stream.py`
  compares loss, entropy, per-token loss, accuracy and the hidden-state gradient with the
  GPU-resident head.

Patches of library internals this needs, each changing where a tensor lives and nothing else:
- `bitsandbytes.nn.Params4bit._quantize`: packed bytes to host RAM after quantizing.
- `bitsandbytes.matmul_4bit` (outermost wrapper): a host weight is handed on as its GPU slot.
- `transformers.core_model_loading.get_device`: `embed_tokens.weight` is loaded to the CPU.
- `transformers.modeling_utils.caching_allocator_warmup`: skipped. It reserves the whole model's
  size on the GPU before loading to speed up loading; most of that is now host RAM.
- `transformers.PreTrainedModel.device`: skips our host-RAM parameters (the model computes on the
  GPU; the Trainer moves inputs to `args.device` either way).
- `peft.prepare_model_for_kbit_training`: keeps the tied embedding in fp16 (above), then builds
  the layer store. `peft.get_peft_model`: peft places each new adapter on its base layer's device,
  host RAM here, so the adapter is moved to the GPU before the optimizer is created.
- `Module._apply` of the modules whose weights we keep in host RAM: `model.to(cuda)` (accelerate)
  converts everything else as usual but leaves those weights where they are.

What it costs: the second forward of every layer (about 10 s of the 64 steps) and the PCIe
traffic it cannot hide (a few seconds): 45.9 s → 63–72 s, depending on the session.

## Our kernels (check 4: 9.2% of the training time)

| Kernel | File | Replaces |
|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | bitsandbytes `kDequantizeBlockwise` (+ absmax rebuild), forward, recompute and backward of every `Linear4bit` |
| `head_chunk_stats_kernel` | `solution/head_stream.py` | softmax / CE / entropy / argmax over the logits, one vocabulary piece at a time |
| `head_chunk_grad_kernel` | `solution/head_stream.py` | the logits gradient `softmax − onehot`, one vocabulary piece at a time |
| `_rmsnorm_backward_input` | `solution/rmsnorm.py` | PyTorch RMSNorm backward |
| `_rmsnorm_forward` | `solution/rmsnorm.py` | PyTorch RMSNorm forward |
| `fused_head_ce_kernel` | `solution/head.py` | the same as the two head kernels, with the head weight on the GPU (used when the offload is off) |

Together they take 9.2% of the training time in the measured run (check 4, steps 3+).

## How to run

On Kaggle (Accelerator **GPU T4 x2**, Internet on), open the public notebook above and press
**Run All**. The repository is the hackathon's private one: first add a Kaggle secret
`GITHUB_TOKEN` (Add-ons → Secrets) with read access to it. The clone cell uses it and never prints
it. Nothing else needs to be set or edited. It clones this repository at the submitted commit, installs the pinned packages,
runs our unit tests (`pytest solution`), trains base Soup and our solution, scores both adapters
and runs the rule checks. It takes about 10 minutes.
Do not run cells that use CUDA in the notebook process itself before the runs: the peak is
measured over the whole GPU, so a CUDA context held by the notebook kernel (~100 MiB) is counted.

Locally on a Linux machine with a CUDA GPU (only a T4 gives the official numbers):

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q          # our kernels and patches against the reference paths
python harness/run_task.py            # base, then solution -> runs/result.json
python harness/check_rules.py         # the 10 checks -> runs/checks.json
python harness/score.py runs/result.json
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q   # Soup's regression tests with our patches
```

`solution.apply()` takes no arguments and installs, in order: the startup patches, RMSNorm, the fused head (model and
trainer sides), the NF4 dequantization, the example SwiGLU, the fp16 MLP, MLP unpadding, the
packed decoder layers, the offload (section 8) and the compile warm-up. `ENABLED = False` in
`offload.py` gives the faster, larger variant (45.9 s, 4,655 MiB). To switch one off, comment out its line in
`solution/__init__.py`. The patches are independent, except that `patch_qwen2_unpad()` and
`patch_qwen2_packed_layers()` must stay after the MLP patches.

## How we got here (Kaggle T4, each row a full `run_task.py`)

| Step | Speed | Memory |
|---|---|---|
| Starter kit (example SwiGLU only) | 0.95x | 1.16x |
| + fused head/loss, MLP unpadding, RMSNorm | 1.29x | 1.95x |
| + NF4 dequantization kernel | 1.24x | 1.93x |
| + MLP in fp16 | 1.28x | 2.41x |
| + fp16 tied embedding, compile warm-up | 1.30x | 2.83x |
| + packed decoder layers | 1.38x | 2.75x |
| + varlen attention on the packed tokens | **1.67x** (base 90 s; ~1.45x estimated for a 77–80 s base) | **3.08x** |
| + startup patches | 1.63x (base 75 s) | 3.08x |
| *option: + recompute every layer (`CHECKPOINT_LAYERS`, off)* | *1.19x* | *6.30x* |
| + offload: NF4 weights, layer inputs, tied embedding in host RAM, streamed head | **1.24x** | **21.4x** (676 MiB) |

Base Soup's wall time varies between Kaggle sessions (76–90 s), and the speed ratio moves with
it. The training loop of our solution varies much less: 34.4 → 30.6 s from varlen attention.

## Limits and what we tried

- **Speed for memory.** The offload costs a second forward of every layer and some PCIe time:
  45.9 s → 63–72 s. By the scoring formula (log of each ratio, relative to the best team) the
  move from 3.08x to ~21x memory is worth more than the speed it costs, so we submit it on.
  `ENABLED = False` in `offload.py` restores the faster variant. The earlier
  `CHECKPOINT_LAYERS` switch in `packed.py` (recompute without the offload, 2,277 MiB) is
  superseded by it and stays off.
- **The step time varies more with the offload** (0.54–1.07 s per step) than without it: long
  batches move more bytes, and the head's two passes over the streamed vocabulary are close to
  the PCIe limit.
- **Recomputing the MLP was turned off.** `mlp_recompute.py` keeps only the fp16 MLP input and
  recomputes gate/up in backward, reusing one dequantization per weight. It is bit-identical and
  brings memory to **4,701 MiB (3.05x)**, but costs about 15% of the training loop (speed
  1.28x → 1.19x). With the scoring formula that trade did not pay off for us, so it is off
  (commented out in `__init__.py`).
- **A Triton GEMM for the head was dropped.** We tried doing the head matmuls with `tl.dot` to
  give our kernels more of the time. On the T4 with Triton 3.6 it was far slower than cuBLAS, so
  the head matmuls stay in cuBLAS.
- **The last bit is not always the same.** Patches 1–4 are bit-identical, or differ only in
  accumulation order. Since the packed layers and varlen attention, the q/k/v/o and LoRA matmuls
  have fewer rows and attention sees other shapes, so the last fp16 bit can differ. Within the
  checks: loss per step up to 0.11% (check 6: first step 0.02%, mean 0.01%), entropy up to
  0.04%, token accuracy up to 0.75% (limit 3%), eval loss −0.000%.
- **The patches are narrow.** They assume Qwen2, bitsandbytes NF4 with fp16 compute, fp16
  autocast, the pinned package versions and right padding. Anything else takes the original
  path, which is correct but slower.
- **Step 1 is about 2.5 s slower** than the median: absmax cache, cuBLAS start-up, the first
  layer and head copies, and whatever compilation the model load did not hide.
- **The memory peak moves a little between runs.** The nvidia-smi peak of one commit varied by
  about ±120 MiB in our runs.
- **The example SwiGLU still runs.** The starter kit's kernel is in the MLP. A kernel of our own
  in its place, fused with the fp16 casts, would be the next step.

## Files

| File | What it is |
|---|---|
| `solution/__init__.py` | `apply()`: installs the patches |
| `solution/head.py` | fused head + loss, Triton kernel, trainer patch, fp16 tied embedding; hands a host-RAM weight to `head_stream.py` |
| `solution/nf4.py` | Triton NF4 dequantization, `bnb.matmul_4bit` patch |
| `solution/rmsnorm.py` | Triton RMSNorm forward and backward |
| `solution/unpad.py` | token index from `attention_mask`, MLP unpadding, final scatter |
| `solution/packed.py` | decoder layers on packed tokens, varlen attention, layer recompute switch |
| `solution/mlp_fp16.py` | MLP in fp16 end to end |
| `solution/mlp_recompute.py` | MLP recomputation (off, see Limits) |
| `solution/warmup.py` | compiles our kernels during model loading |
| `solution/startup.py` | the two startup patches (section 7) |
| `solution/offload.py` | NF4 weights and layer inputs in pinned host RAM, streamed; recompute; tied embedding in host RAM (section 8) |
| `solution/head_stream.py` | the fused head with the vocabulary streamed from host RAM, two Triton kernels (section 8) |
| `solution/swiglu.py` | the starter kit's example SwiGLU (not ours, not counted) |
| `solution/test_*.py` | 56 tests: each patch against its reference path, plus a 40-step convergence test |

## Credits

- **Streaming frozen weights from host RAM, layer by layer:** the general idea of CPU offloading
  (e.g. ZeRO-Offload, [arXiv:2101.06840](https://arxiv.org/abs/2101.06840)); our code is our own.
- **Fused linear cross-entropy in chunks:** idea from Liger Kernel
  ([arXiv:2410.10989](https://arxiv.org/abs/2410.10989)).
- **Online softmax statistics:** Milakov & Gimelshein (2018); Cut Your Losses
  ([arXiv:2411.09009](https://arxiv.org/abs/2411.09009)).
- **NF4 table and arithmetic:** from bitsandbytes `csrc/kernels.cu` (MIT), so that our
  dequantization matches it bit for bit.
- **Rewind contract:** the no-logits contract (`_soup_chunked_step`) is Soup's own (#1693).

The code in `solution/` is ours, apart from the starter kit's `swiglu.py`. Starter kit, harness
and tests: Apache-2.0, see `LICENSE` and `NOTICE`. Data:
[AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2).
Model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).