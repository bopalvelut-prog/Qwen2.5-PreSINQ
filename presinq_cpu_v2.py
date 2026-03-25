"""
Memory-efficient Pre-SINQ for Qwen2.5 models.
Processes layer-by-layer with explicit memory management.
"""
import os
import sys
import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sinq.sinkhorn import sinkhorn_log


def find_block(H, W, block):
    for i in range(W):
        if (W % (block + i) == 0):
            return block + i
        elif (W % (block - i) == 0):
            return block - i
    return block


def get_sink_scale_small(tensors, block=64, n_iter=4):
    """Compute Sinkhorn scales with minimal memory."""
    W = torch.cat(tensors, dim=0).float()
    H, Wid = W.shape
    
    if block <= 0:
        W_hat, mu1, mu2 = sinkhorn_log(W, n_iter)
    else:
        if Wid % block != 0:
            block = find_block(H, Wid, block)
        n_w = Wid // block
        W = W.view(H, n_w, block).permute(1, 0, 2).reshape(n_w, H * block)
        
        scales = []
        for i in range(n_w):
            block_mat = W[i].view(H, block)
            _, mu1, _ = sinkhorn_log(block_mat, n_iter)
            scales.append(mu1 / mu1.median())
        mu1 = torch.stack(scales).view(-1)
    
    del W
    gc.collect()
    return mu1.to(torch.float16)


def get_sink_scale_inplace(tensors, block=64, n_iter=4):
    """Compute Sinkhorn scales per output channel."""
    shapes = [t.shape for t in tensors]
    total_out = sum(s[0] for s in shapes)
    Wid = shapes[0][1]
    dtype = tensors[0].dtype
    
    W_cat = torch.zeros(total_out, Wid, dtype=torch.float32)
    offset = 0
    for t in tensors:
        sz = t.shape[0]
        W_cat[offset:offset+sz] = t.float()
        offset += sz
    
    H, Wid = W_cat.shape
    
    if block <= 0:
        W_hat, mu1, mu2 = sinkhorn_log(W_cat, n_iter)
    else:
        if Wid % block != 0:
            block = find_block(H, Wid, block)
        n_w = Wid // block
        
        scales_list = []
        for w in range(n_w):
            start_col = w * block
            end_col = start_col + block
            block_mat = W_cat[:, start_col:end_col]
            _, mu1, _ = sinkhorn_log(block_mat, n_iter)
            scales_list.append((mu1 / mu1.median()).to(dtype))
        mu1 = torch.cat(scales_list)
    
    del W_cat
    gc.collect()
    return mu1.to(dtype)


def apply_layer_presinq(layer, group_size=64, n_iter=4):
    """Apply Pre-SINQ to a single layer."""
    dtype = layer.input_layernorm.weight.dtype
    
    # 1. QKV normalization
    qkv_tensors = [
        layer.self_attn.q_proj.weight.data,
        layer.self_attn.k_proj.weight.data,
        layer.self_attn.v_proj.weight.data
    ]
    t_qkv = get_sink_scale_inplace(qkv_tensors, block=group_size, n_iter=n_iter)
    
    layer.input_layernorm.weight.data = (layer.input_layernorm.weight.data * t_qkv).to(dtype)
    
    for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        proj.weight.data = torch.matmul(proj.weight.data, torch.diag(1 / t_qkv)).to(dtype)
    
    del qkv_tensors, t_qkv
    gc.collect()
    
    # 2. Gate+Up normalization
    gu_tensors = [
        layer.mlp.gate_proj.weight.data,
        layer.mlp.up_proj.weight.data
    ]
    t_gu = get_sink_scale_inplace(gu_tensors, block=group_size, n_iter=n_iter)
    
    layer.post_attention_layernorm.weight.data = (layer.post_attention_layernorm.weight.data * t_gu).to(dtype)
    
    layer.mlp.gate_proj.weight.data = torch.matmul(layer.mlp.gate_proj.weight.data, torch.diag(1 / t_gu)).to(dtype)
    layer.mlp.up_proj.weight.data = torch.matmul(layer.mlp.up_proj.weight.data, torch.diag(1 / t_gu)).to(dtype)
    
    del gu_tensors, t_gu
    gc.collect()
    
    # 3. Down projection normalization
    t_d = get_sink_scale_inplace([layer.mlp.down_proj.weight.data], block=group_size, n_iter=n_iter)
    
    layer.mlp.down_proj.weight.data = torch.matmul(layer.mlp.down_proj.weight.data, torch.diag(1 / t_d)).to(dtype)
    layer.mlp.up_proj.weight.data = torch.matmul(torch.diag(t_d), layer.mlp.up_proj.weight.data).to(dtype)
    
    del t_d
    gc.collect()
    
    # 4. Output projection (GQA)
    n_group = layer.self_attn.v_proj.weight.shape[0] // layer.self_attn.o_proj.weight.shape[0]
    if n_group > 0:
        oOut, oIn = layer.self_attn.o_proj.weight.shape
        o_weight = layer.self_attn.o_proj.weight.data.reshape(n_group * oOut, -1)
        t_o = get_sink_scale_inplace([o_weight], block=group_size, n_iter=n_iter)
        
        layer.self_attn.v_proj.weight.data = torch.matmul(torch.diag(t_o), layer.self_attn.v_proj.weight.data).to(dtype)
        
        t_o_cat = torch.cat([t_o] * n_group)
        layer.self_attn.o_proj.weight.data = torch.matmul(
            layer.self_attn.o_proj.weight.data, torch.diag(1 / t_o_cat)
        ).to(dtype)
        
        del o_weight, t_o, t_o_cat
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient Pre-SINQ for Qwen2.5")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default="./qwen25-0.5b-presinq")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--n_iter", type=int, default=4)
    parser.add_argument("--n_repeat", type=int, default=3)
    args = parser.parse_args()

    print(f"Model: {args.model_name}")
    print(f"Config: group_size={args.group_size}, n_iter={args.n_iter}, n_repeat={args.n_repeat}")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    
    print("\nLoading model (float16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    print(f"Total layers: {len(model.model.layers)}")
    
    # Process each repeat
    for r in range(args.n_repeat):
        print(f"\n--- Repeat {r+1}/{args.n_repeat} ---")
        for layer_idx in tqdm(range(len(model.model.layers)), desc="Layers"):
            layer = model.model.layers[layer_idx]
            apply_layer_presinq(layer, args.group_size, args.n_iter)
            gc.collect()
    
    print("\nSaving model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✓ Saved to {args.output_dir}")
    print("\nTo convert to GGUF:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {args.output_dir} --outfile model-f16.gguf --outtype f16")


if __name__ == "__main__":
    main()
