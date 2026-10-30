# Submission: <TEAM_NAME>

## Links
- Repository, final commit: <COMMIT_LINK>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  our README): [<KAGGLE_NOTEBOOK_LINK>](https://www.kaggle.com/code/djparty/soupv3/edit)
- Demo video (up to 3 minutes): <VIDEO_LINK>
- Presentation: <PRESENTATION_LINK>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Adil | Optimization Lead / Team Captain | https://github.com/Adul23/ | https://www.kaggle.com/djparty | @yert24(TG) |
| Aidar | GPU Performance Engineer | https://github.com/aidarBARMEN | https://www.kaggle.com/aidarBARMEN | @johncena66(TG) |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | bitsandbytes NF4 `kDequantizeBlockwise` + absmax rebuild, forward and backward of every `Linear4bit` | 6.5% |
| `_rmsnorm_backward_input` | `solution/rmsnorm.py` | PyTorch `Qwen2RMSNorm` backward | 0.7% |
| `fused_head_ce_kernel` | `solution/head.py` | softmax, cross-entropy, entropy and argmax over the full `[B, T, 151936]` logits | 0.7% |
| `_rmsnorm_forward` | `solution/rmsnorm.py` | PyTorch `Qwen2RMSNorm` forward | 0.5% |
| **Total** | | | **8.4%** |

The starter kit's example SwiGLU kernels still run (`solution/swiglu.py`). They are not ours, and
we do not count them.

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Chunked fused linear cross-entropy: Liger Kernel, https://arxiv.org/abs/2410.10989
- Online softmax statistics over the vocabulary: Cut Your Losses, https://arxiv.org/abs/2411.09009
- NF4 lookup table and dequantization arithmetic (to match it bit for bit): bitsandbytes
  `csrc/kernels.cu`, https://github.com/bitsandbytes-foundation/bitsandbytes

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 80.04 | 58.34 | speedup 1.372x |
| Peak GPU memory, MiB | 14,345 | 5,217 | base/ours 2.750x |
| Eval loss (public rows) | 2.04961 | 2.04965 | +0.002% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed (check 4: 8.4%; check 6: first step 0.02%,
  mean 0.01%; check 7: +0.002%)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
- **What we sped up and why:** the full-vocabulary logits and their fp32 copies, NF4
  dequantization, fp32 copies of fp16 activations, and work on padding.
- **How the kernels work:**
  - the fused head (chunked logits, one-pass online log-sum-exp / entropy / argmax, in-place
    gradient);
  - NF4 dequantization, bit-identical to bitsandbytes;
  - RMSNorm.
- **Supporting changes:** MLP in fp16, packed decoder layers with LoRA dropout kept on the base
  RNG stream, and compiling the kernels during model load.
- **Results against the base:** 1.37x speed, 2.75x memory, eval loss +0.002%, and how each step
  contributed.
- **Limits and next steps:** see below.

## Limitations
- **MLP recomputation is off.** `solution/mlp_recompute.py` reaches 4,701 MiB (3.05x) but costs
  about 15% of the training loop.
- **SDPA still runs on the padded batch.** A varlen attention kernel is the next speed step.
- **A Triton GEMM for the head was slower than cuBLAS** on the T4, so the head matmuls stay in
  cuBLAS.
- **The last bit is not always the same.** Since the packed layers, a different cuBLAS algorithm
  for the shorter matmuls can change the last fp16 bit. All checks stay well inside their limits.
- **The patches are narrow.** They assume Qwen2 + bitsandbytes NF4 (fp16 compute) + fp16 autocast
  and the pinned versions. Elsewhere they fall back to the original code.
- **Step 1 is still about 1.9 s slower** than the median: absmax cache, embedding check, CUDA
  start-up.