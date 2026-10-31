# Soup Kernel Challenge: 1.45–1.67x faster, 3.08x less GPU memory on a T4

Our solution makes `soup train` (Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4 + LoRA, 64 steps on a
Kaggle T4) **1.45–1.67x faster** and cuts its peak GPU memory **from 14,345 MiB to 4,657 MiB
(3.08x)**. Eval loss stays the same (−0.000%), and all 10 checks of `harness/check_rules.py` pass.
Four `@triton.jit` kernels of our own take 10.1% of the training time. Everything lives in
`solution/` and is installed by `solution.apply()`.

The speed ratio depends on how fast base Soup happens to run in the session. In the run below
base Soup took 90.4 s, which gave 1.67x. In our other sessions it took 77–80 s; against that, our
54 s would be about 1.45x (an estimate, not a measured run of this commit). The training loop
alone is steadier: 1.83x. An optional switch trades speed for
memory: **2,277 MiB (6.3x less)** at about 1.19x (see Limits).

**Public Kaggle notebook (the measured run):** https://www.kaggle.com/code/djparty/soupv4  
**Final commit:** https://github.com/Adul23/soup-kernel-challenge/commit/<FINAL_SHA>

| Kaggle T4, `harness/run_task.py` | Base Soup | Ours | Ratio |
|---|---|---|---|
| Wall time of `soup train`, s | 90.42 | 54.06 | **1.673x faster** |
| Training loop (64 steps), s | 56.01 | 30.57 | 1.83x |
| Peak GPU memory (nvidia-smi), MiB | 14,345 | 4,657 | **3.080x less** |
| Eval loss, public rows | 2.04965 | 2.04964 | −0.000% |
| Loss curve vs base (check 6) | | first step 0.02%, mean 0.01% | limits 1% / 3% |

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
- **Chunked logits.** cuBLAS computes the logits for 512 tokens at a time into one reused fp16
  buffer. The fp16 inputs, fp32 accumulation and fp16 output are what autocast does.
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
  value. If any value is not fp16-exact, the weight is left as it is.

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
checks this by comparing CUDA RNG states.

### 6. Kernel compilation off the critical path (`warmup.py`)

Every measured run starts with empty Triton caches. Compiling our kernels on the first launches
added about 4 s to step 1. A thread inside the same `soup train` process compiles them
(`JITFunction.warmup`) while Soup loads and quantizes the model, and the trainer waits for it
before step 1. It is the same compilation, timed inside the same command, only earlier. Nothing
is cached across runs. Setting `ENABLED = False` in `warmup.py` restores compile-on-first-use.

## Our kernels (check 4: 10.1% of the training time)

| Kernel | File | Replaces | ms in the measured run (steps 3+) | Share |
|---|---|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | bitsandbytes `kDequantizeBlockwise` (+ absmax rebuild) | 2,117 | 7.8% |
| `_rmsnorm_backward_input` | `solution/rmsnorm.py` | PyTorch RMSNorm backward | 234 | 0.9% |
| `fused_head_ce_kernel` | `solution/head.py` | softmax / CE / entropy / argmax over the full logits | 197 | 0.7% |
| `_rmsnorm_forward` | `solution/rmsnorm.py` | PyTorch RMSNorm forward | 183 | 0.7% |

## How to run

On Kaggle (Accelerator **GPU T4 x2**, Internet on), open the public notebook above and press
**Run All**. It clones this repository at the submitted commit, installs the pinned packages,
runs our unit tests (`pytest solution`), trains base Soup and our solution, scores both adapters
and runs the rule checks. It takes about 10 minutes.

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

`solution.apply()` takes no arguments and installs, in order: RMSNorm, the fused head (model and
trainer sides), the NF4 dequantization, the example SwiGLU, the fp16 MLP, MLP unpadding, the
packed decoder layers and the compile warm-up. To switch one off, comment out its line in
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
| *option: + recompute every layer (`CHECKPOINT_LAYERS`, off)* | *1.19x* | *6.30x* |

Base Soup's wall time varies between Kaggle sessions (76–90 s), and the speed ratio moves with
it. The training loop of our solution varies much less: 34.4 → 30.6 s from varlen attention.

## Limits and what we tried

- **Recomputing every layer is a switch, and it is off.** With `CHECKPOINT_LAYERS = True` in
  `packed.py`, each packed layer runs under our own non-reentrant `torch.utils.checkpoint`
  (RNG preserved, so LoRA dropout replays the same masks), and only its `[n_real, 1536]` input
  is kept for backward. Soup's `gradient_checkpointing` switch stays off. Peak memory drops to
  **2,277 MiB (6.30x less than base Soup)**, below the ~2,400 MiB the task quotes for Unsloth,
  but the second forward costs about 18% of the speed (1.19x with a 77 s base). By the scoring
  formula the faster setting comes out ahead unless other teams are both much faster and much
  lighter, so we submit it off. All checks pass with it on too.
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
- **Step 1 is still about 1.9 s slower** than the median: absmax cache, embedding check, cuBLAS
  start-up, and whatever compilation the model load did not hide.
- **The memory peak moves a little between runs.** The nvidia-smi peak of one commit varied by
  about ±120 MiB in our runs.
- **The example SwiGLU still runs.** The starter kit's kernel is in the MLP. A kernel of our own
  in its place, fused with the fp16 casts, would be the next step.

## Files

| File | What it is |
|---|---|
| `solution/__init__.py` | `apply()`: installs the patches |
| `solution/head.py` | fused head + loss, Triton kernel, trainer patch, fp16 tied embedding |
| `solution/nf4.py` | Triton NF4 dequantization, `bnb.matmul_4bit` patch |
| `solution/rmsnorm.py` | Triton RMSNorm forward and backward |
| `solution/unpad.py` | token index from `attention_mask`, MLP unpadding, final scatter |
| `solution/packed.py` | decoder layers on packed tokens, varlen attention, layer recompute switch |
| `solution/mlp_fp16.py` | MLP in fp16 end to end |
| `solution/mlp_recompute.py` | MLP recomputation (off, see Limits) |
| `solution/warmup.py` | compiles our kernels during model loading |
| `solution/swiglu.py` | the starter kit's example SwiGLU (not ours, not counted) |
| `solution/test_*.py` | 48 tests: each patch against its reference path |

## Credits

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
