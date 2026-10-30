"""Your solution.

The harness imports this package and calls `apply()` (no arguments) once, in the same process as
`soup train`, before Soup builds the model and the trainer. All patches are class-level, so they
reach the model and the trainer Soup builds later, and each one falls back to the original code
outside the exact situation it was written for.
"""

from .head import patch_fused_head
from .mlp_fp16 import patch_qwen2_mlp_fp16
from .mlp_recompute import patch_qwen2_mlp_recompute
from .nf4 import patch_bnb_matmul_4bit
from .rmsnorm import patch_qwen2_rmsnorm
from .swiglu import patch_qwen2_mlp
from .unpad import patch_qwen2_unpad


def apply(model=None):  # the harness calls apply(); `model` is accepted only for convenience
    patch_qwen2_rmsnorm()  # Triton RMSNorm (ours)
    patch_fused_head()     # Triton fused head + loss (ours): no full logits tensor
    patch_bnb_matmul_4bit()  # Triton NF4 weight dequantization (ours), bit-identical
    patch_qwen2_mlp()      # example SwiGLU (about -14% peak memory); does not count as ours
    patch_qwen2_mlp_fp16()  # MLP activations in fp16 instead of fp32 copies, same values
    # patch_qwen2_mlp_recompute()  # 3.05x memory but -7.5% speed: off (see README)
    patch_qwen2_unpad()    # keep last: it wraps whatever Qwen2MLP.forward is installed by now
