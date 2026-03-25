"""
CPU-compatible Pre-SINQ for Qwen2.5 models.
Based on huawei-csl/SINQ but modified to run without CUDA.
"""

import os
import sys
import argparse
import gc

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add SINQ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sinq.sinkhorn import sinkhorn_log


def find_block(H, W, block):
    for i in range(W):
        if W % (block + i) == 0:
            return block + i
        elif W % (block - i) == 0:
            return block - i
    return block


def get_sink_scale(matrix_list, cat_dim=0, block=64, n_iter=4):
    """Compute Sinkhorn scales on CPU."""
    W = torch.cat(matrix_list, dim=cat_dim)
    H, Wid = W.shape
    dtype = W.dtype
    W = W.float()

    if block <= 0:
        W_hat, mu1, mu2 = sinkhorn_log(W, n_iter)
    else:
        if Wid % block != 0:
            block = find_block(H, Wid, block)
        assert Wid % block == 0, "block must divide W"
        n_w = Wid // block
        W = W.view(H, Wid // block, block)
        W_batched = W.permute(1, 0, 2).contiguous().view(n_w, H, block)

        def process_block(mat):
            return sinkhorn_log(mat, n_iter)

        W_hat, mu1, mu2 = torch.vmap(process_block, randomness="different")(W_batched)

    mu1 = mu1 / mu1.median()
    return mu1.view(-1).to(dtype)


def get_core_layers_module(model):
    core = model
    for _ in range(4):
        if hasattr(core, "layers"):
            return core
        if hasattr(core, "model"):
            core = core.model
        else:
            break
    raise AttributeError("Could not find `.layers` module")


def get_layer_prefix(model):
    core = model
    parts = []
    for _ in range(4):
        if hasattr(core, "layers"):
            parts.append("layers")
            return ".".join(parts[:-1]) if len(parts) > 1 else "model.layers"
        if hasattr(core, "model"):
            parts.append("model")
            core = core.model
        else:
            break
    return "model.layers"


def pre_sinq_qwen(model, n_repeat=3, group_size=64, n_iter=4):
    """
    Apply Pre-SINQ (Sinkhorn normalization) to Qwen2.5/3 models.
    This is the core algorithm without AWQ - just weight reparameterization.
    """
    core = get_core_layers_module(model)

    for _ in range(n_repeat):
        for layer_idx, layer in tqdm(enumerate(core.layers), desc="Pre-SINQ (Qwen)"):
            dev = layer.input_layernorm.weight.device
            dtype = layer.input_layernorm.weight.dtype

            # 1. Attention QKV normalization
            t_qkv = get_sink_scale(
                [
                    layer.self_attn.q_proj.weight.data,
                    layer.self_attn.k_proj.weight.data,
                    layer.self_attn.v_proj.weight.data,
                ],
                block=group_size,
                n_iter=n_iter,
            )
            # Absorb into input_layernorm
            layer.input_layernorm.weight.data = (
                layer.input_layernorm.weight.data * t_qkv.view(-1)
            ).to(dev)
            # Scale down Q, K, V projections
            layer.self_attn.q_proj.weight.data = torch.matmul(
                layer.self_attn.q_proj.weight.data, torch.diag(1 / t_qkv)
            ).to(dev)
            layer.self_attn.k_proj.weight.data = torch.matmul(
                layer.self_attn.k_proj.weight.data, torch.diag(1 / t_qkv)
            ).to(dev)
            layer.self_attn.v_proj.weight.data = torch.matmul(
                layer.self_attn.v_proj.weight.data, torch.diag(1 / t_qkv)
            ).to(dev)

            # 2. MLP gate+up normalization
            t_gu = get_sink_scale(
                [layer.mlp.gate_proj.weight.data, layer.mlp.up_proj.weight.data],
                block=group_size,
                n_iter=n_iter,
            )
            # Absorb into post_attention_layernorm
            layer.post_attention_layernorm.weight.data = (
                layer.post_attention_layernorm.weight.data * t_gu.view(-1)
            ).to(dev)
            # Scale down gate and up projections
            layer.mlp.gate_proj.weight.data = torch.matmul(
                layer.mlp.gate_proj.weight.data, torch.diag(1 / t_gu)
            ).to(dev)
            layer.mlp.up_proj.weight.data = torch.matmul(
                layer.mlp.up_proj.weight.data, torch.diag(1 / t_gu)
            ).to(dev)

            # 3. MLP down normalization
            t_d = get_sink_scale(
                [layer.mlp.down_proj.weight.data], block=group_size, n_iter=n_iter
            )
            # Scale down down_proj, scale up up_proj
            layer.mlp.down_proj.weight.data = torch.matmul(
                layer.mlp.down_proj.weight.data, torch.diag(1 / t_d)
            ).to(dev)
            layer.mlp.up_proj.weight.data = torch.matmul(
                torch.diag(t_d), layer.mlp.up_proj.weight.data
            ).to(dev)

            # 4. Optional: Attention output projection normalization
            # This handles GQA (grouped query attention) for Qwen models
            n_group = (
                layer.self_attn.v_proj.weight.shape[0]
                // layer.self_attn.o_proj.weight.shape[0]
            )
            if n_group > 0:
                oOut, oIn = layer.self_attn.o_proj.weight.shape
                t_o = get_sink_scale(
                    [layer.self_attn.o_proj.weight.data.reshape(n_group * oOut, -1)],
                    block=group_size,
                    n_iter=n_iter,
                )
                # Scale v_proj
                layer.self_attn.v_proj.weight.data = torch.matmul(
                    torch.diag(t_o), layer.self_attn.v_proj.weight.data
                ).to(dev)
                # Scale o_proj
                t_o_cat = torch.cat([t_o] * n_group)
                layer.self_attn.o_proj.weight.data = torch.matmul(
                    layer.self_attn.o_proj.weight.data, torch.diag(1 / t_o_cat)
                ).to(dev)

    return model


def main():
    parser = argparse.ArgumentParser(description="Pre-SINQ for Qwen2.5 (CPU mode)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default="./qwen25-0.5b-presinq")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--n_iter", type=int, default=4)
    parser.add_argument("--n_repeat", type=int, default=3)
    parser.add_argument("--skip_gguf", action="store_true", help="Skip GGUF conversion")
    args = parser.parse_args()

    device = "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(
        f"Pre-SINQ config: group_size={args.group_size}, n_iter={args.n_iter}, n_repeat={args.n_repeat}"
    )

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for CPU stability
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    model.eval()

    # Apply Pre-SINQ
    print("\nApplying Pre-SINQ...")
    model = pre_sinq_qwen(
        model, n_repeat=args.n_repeat, group_size=args.group_size, n_iter=args.n_iter
    )

    # Save the Pre-SINQ model in float16 to save RAM
    print(f"\nSaving Pre-SINQ model to {args.output_dir} (float16)...")
    gc.collect()
    os.makedirs(args.output_dir, exist_ok=True)
    model = model.to(dtype=torch.float16)
    gc.collect()
    model.save_pretrained(args.output_dir, max_shard_size="500MB")
    tokenizer.save_pretrained(args.output_dir)
    print(f"Pre-SINQ model saved to {args.output_dir}")

    if not args.skip_gguf:
        print("\n" + "=" * 60)
        print("GGUF Conversion Instructions")
        print("=" * 60)
        print(f"""
To convert to GGUF, use llama.cpp:

1. Clone llama.cpp (if not already):
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp && make -j

2. Convert to GGUF FP16:
   python llama.cpp/convert_hf_to_gguf.py {args.output_dir} \\
     --outfile qwen25-0.5b-presinq-f16.gguf --outtype f16

3. Quantize to Q4_K_M:
   ./llama.cpp/llama-quantize qwen25-0.5b-presinq-f16.gguf \\
     qwen25-0.5b-presinq-Q4_K_M.gguf Q4_K_M
""")

    print("\n✓ Pre-SINQ complete!")


if __name__ == "__main__":
    main()
