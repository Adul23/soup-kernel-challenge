"""Streamed NF4 weights + recomputed layers give the same loss, grads and RNG as the plain run."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)


def _model():
    import bitsandbytes as bnb
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(cfg).half()
    for layer in model.model.layers:
        for parent, names in (
            (layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
            (layer.mlp, ("gate_proj", "up_proj", "down_proj")),
        ):
            for name in names:
                lin = getattr(parent, name)
                q = bnb.nn.Linear4bit(
                    lin.in_features,
                    lin.out_features,
                    bias=lin.bias is not None,
                    compute_dtype=torch.float16,
                    quant_type="nf4",
                    compress_statistics=True,
                )
                q.weight = bnb.nn.Params4bit(
                    lin.weight.data.clone(),
                    requires_grad=False,
                    quant_type="nf4",
                    compress_statistics=True,
                )
                if lin.bias is not None:
                    q.bias = torch.nn.Parameter(
                        lin.bias.data.clone().float(), requires_grad=False
                    )
                setattr(parent, name, q)
    model = model.cuda()  # quantizes (the offload hook may move the bytes to host RAM)
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data.float()
    for norm in [m for m in model.modules() if "RMSNorm" in type(m).__name__]:
        norm.weight.data = norm.weight.data.float()
    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05)
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.q_proj = LoraLinear4bit(
            attn.q_proj, "default", config=config, r=4, lora_alpha=8
        )
        attn.v_proj = LoraLinear4bit(
            attn.v_proj, "default", config=config, r=4, lora_alpha=8
        )
    dev = torch.device("cuda")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.data = p.data.float().to(dev)
                p.requires_grad_(True)
                if "lora_B" in name:
                    p.normal_(0.0, 0.05)
    return model.train()


def test_streamed_layers_match():
    from solution import nf4, offload, packed, unpad

    unpad.patch_qwen2_unpad()
    packed.patch_qwen2_packed_layers()
    nf4.patch_bnb_matmul_4bit()
    offload.patch_matmul_4bit()

    ids = torch.randint(0, 1000, (4, 40), device="cuda")
    mask = torch.ones_like(ids)
    for row, n in enumerate([40, 23, 9, 31]):
        mask[row, n:] = 0
    labels = ids.masked_fill(mask == 0, -100)

    def run(stream):
        offload._STORE["store"] = None
        model = _model()
        if stream:
            store = offload.build_store(model)
            assert store is not None
            assert all(
                not m.weight.is_cuda
                for m in model.modules()
                if offload._is_nf4_linear(m)
            )
        try:
            torch.manual_seed(1234)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(
                    input_ids=ids, attention_mask=mask, labels=labels, use_cache=False
                )
            out.loss.backward()
        finally:
            offload._STORE["store"] = None
        grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        return out.loss.detach(), grads, torch.cuda.get_rng_state()

    loss_ref, grads_ref, rng_ref = run(False)
    for _ in range(2):  # twice: the slots are reused across steps
        loss, grads, rng = run(True)
        assert torch.equal(rng, rng_ref), "the recompute changed the RNG stream"
        torch.testing.assert_close(loss, loss_ref, rtol=1e-4, atol=1e-5)
        assert grads.keys() == grads_ref.keys() and any("lora_A" in n for n in grads)
        scale = max(g.abs().max().item() for g in grads_ref.values())
        for name, g in grads_ref.items():
            torch.testing.assert_close(
                grads[name], g, rtol=1e-2, atol=1e-3 * scale + 1e-8, msg=name
            )
