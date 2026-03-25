#!/usr/bin/env python3
"""
Pre-SINQ for Qwen3.5 models (GatedDeltaNet / linear attention).
Based on huawei-csl/SINQ, adapted for Qwen3.5 architecture.
"""

import os
import sys
import argparse
import gc

import torch
import torch.nn as nn
from tqdm import tqdm

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


def pre_sinq_qwen35(model, n_repeat=3, group_size=64, n_iter=4):
    """Pre-SINQ for Qwen3.5 hybrid (GatedDeltaNet + standard attention)."""
    core = get_core_layers_module(model)

    for _ in range(n_repeat):
        for layer_idx, layer in tqdm(enumerate(core.layers), desc="Pre-SINQ (Qwen3.5)"):
            dev = layer.input_layernorm.weight.device
            dtype = layer.input_layernorm.weight.dtype

            if hasattr(layer, "self_attn"):
                # Standard attention layer (every 4th layer)
                attn = layer.self_attn
                t_qkv = get_sink_scale(
                    [
                        attn.q_proj.weight.data,
                        attn.k_proj.weight.data,
                        attn.v_proj.weight.data,
                    ],
                    block=group_size,
                    n_iter=n_iter,
                )
                layer.input_layernorm.weight.data = (
                    layer.input_layernorm.weight.data * t_qkv.view(-1)
                ).to(dev)
                attn.q_proj.weight.data = torch.matmul(
                    attn.q_proj.weight.data, torch.diag(1 / t_qkv)
                ).to(dev)
                attn.k_proj.weight.data = torch.matmul(
                    attn.k_proj.weight.data, torch.diag(1 / t_qkv)
                ).to(dev)
                attn.v_proj.weight.data = torch.matmul(
                    attn.v_proj.weight.data, torch.diag(1 / t_qkv)
                ).to(dev)

                # Output projection
                n_group = attn.v_proj.weight.shape[0] // attn.o_proj.weight.shape[0]
                if n_group > 0:
                    oOut, oIn = attn.o_proj.weight.shape
                    t_o = get_sink_scale(
                        [attn.o_proj.weight.data.reshape(n_group * oOut, -1)],
                        block=group_size,
                        n_iter=n_iter,
                    )
                    attn.v_proj.weight.data = torch.matmul(
                        torch.diag(t_o), attn.v_proj.weight.data
                    ).to(dev)
                    t_o_cat = torch.cat([t_o] * n_group)
                    attn.o_proj.weight.data = torch.matmul(
                        attn.o_proj.weight.data, torch.diag(1 / t_o_cat)
                    ).to(dev)

            else:
                # Linear attention layer (GatedDeltaNet)
                attn = layer.linear_attn

                # QKV normalization
                t_qkv = get_sink_scale(
                    [attn.in_proj_qkv.weight.data], block=group_size, n_iter=n_iter
                )
                layer.input_layernorm.weight.data = (
                    layer.input_layernorm.weight.data * t_qkv.view(-1)
                ).to(dev)
                attn.in_proj_qkv.weight.data = torch.matmul(
                    attn.in_proj_qkv.weight.data, torch.diag(1 / t_qkv)
                ).to(dev)

                # Gating projection normalization
                t_z = get_sink_scale(
                    [attn.in_proj_z.weight.data], block=group_size, n_iter=n_iter
                )
                layer.input_layernorm.weight.data = (
                    layer.input_layernorm.weight.data * t_z.view(-1)
                ).to(dev)
                attn.in_proj_z.weight.data = torch.matmul(
                    attn.in_proj_z.weight.data, torch.diag(1 / t_z)
                ).to(dev)

                # Output projection
                t_o = get_sink_scale(
                    [attn.out_proj.weight.data], block=group_size, n_iter=n_iter
                )
                attn.out_proj.weight.data = torch.matmul(
                    attn.out_proj.weight.data, torch.diag(1 / t_o)
                ).to(dev)

            # MLP normalization (same for all layers)
            t_gu = get_sink_scale(
                [layer.mlp.gate_proj.weight.data, layer.mlp.up_proj.weight.data],
                block=group_size,
                n_iter=n_iter,
            )
            layer.post_attention_layernorm.weight.data = (
                layer.post_attention_layernorm.weight.data * t_gu.view(-1)
            ).to(dev)
            layer.mlp.gate_proj.weight.data = torch.matmul(
                layer.mlp.gate_proj.weight.data, torch.diag(1 / t_gu)
            ).to(dev)
            layer.mlp.up_proj.weight.data = torch.matmul(
                layer.mlp.up_proj.weight.data, torch.diag(1 / t_gu)
            ).to(dev)

            t_d = get_sink_scale(
                [layer.mlp.down_proj.weight.data], block=group_size, n_iter=n_iter
            )
            layer.mlp.down_proj.weight.data = torch.matmul(
                layer.mlp.down_proj.weight.data, torch.diag(1 / t_d)
            ).to(dev)
            layer.mlp.up_proj.weight.data = torch.matmul(
                torch.diag(t_d), layer.mlp.up_proj.weight.data
            ).to(dev)

    return model


def main():
    parser = argparse.ArgumentParser(description="Pre-SINQ for Qwen3.5 (CPU mode)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output_dir", type=str, default="./qwen35-0.8b-presinq")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--n_iter", type=int, default=4)
    parser.add_argument("--n_repeat", type=int, default=3)
    parser.add_argument("--skip_gguf", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(
        f"Pre-SINQ config: group_size={args.group_size}, n_iter={args.n_iter}, n_repeat={args.n_repeat}"
    )

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    model.eval()

    print("\nApplying Pre-SINQ...")
    model = pre_sinq_qwen35(
        model, n_repeat=args.n_repeat, group_size=args.group_size, n_iter=args.n_iter
    )

    print(f"\nSaving Pre-SINQ model to {args.output_dir} (float16)...")
    gc.collect()
    os.makedirs(args.output_dir, exist_ok=True)
    model = model.to(dtype=torch.float16)
    gc.collect()
    model.save_pretrained(args.output_dir, max_shard_size="500MB")
    tokenizer.save_pretrained(args.output_dir)
    print(f"Pre-SINQ model saved to {args.output_dir}")

    if not args.skip_gguf:
        print("\nTo convert to GGUF:")
        print(
            f"  python ~/prima.cpp/convert_hf_to_gguf.py {args.output_dir} --outfile model-f16.gguf --outtype f16"
        )
        print(
            f"  ~/prima.cpp/build/bin/llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M"
        )

    print("\nPre-SINQ complete!")


if __name__ == "__main__":
    main()
