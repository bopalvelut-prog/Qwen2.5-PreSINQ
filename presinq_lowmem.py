#!/usr/bin/env python3
"""
Memory-optimized Pre-SINQ for Qwen2.5 on low-RAM systems (<4GB).
Uses torch.inference_mode() and processes per-shard to minimize peak RAM.
"""
import os
import sys
import gc
import shutil
import argparse
from pathlib import Path
from typing import Optional

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SINQ"))
from sinq.sinkhorn import sinkhorn_log


torch.set_num_threads(2)
torch.set_num_interop_threads(1)


def find_block(H, W, block):
    for i in range(W):
        if W % (block + i) == 0:
            return block + i
        if W % (block - i) == 0:
            return block - i
    return block


def sink_scale(tensors, block=64, n_iter=4):
    shapes = [t.shape for t in tensors]
    total_out = sum(s[0] for s in shapes)
    Wid = shapes[0][1]
    dtype = tensors[0].dtype
    
    if block <= 0:
        W_cat = torch.cat(tensors, dim=0).float()
        _, mu1, _ = sinkhorn_log(W_cat, n_iter)
        del W_cat
        gc.collect()
        return mu1.to(dtype)
    
    if Wid % block != 0:
        block = find_block(total_out, Wid, block)
    n_w = Wid // block
    
    scales = []
    for w in range(n_w):
        start, end = w * block, (w + 1) * block
        parts = [t.float()[:, start:end] for t in tensors]
        mat = torch.cat(parts, dim=0)
        del parts
        _, mu1, _ = sinkhorn_log(mat, n_iter)
        scales.append((mu1 / mu1.median()).to(dtype))
        del mat
        gc.collect()
    
    return torch.cat(scales)


def apply_presinq_layer(layer_state):
    dtype = layer_state["q_proj"].dtype
    
    t_qkv = sink_scale([layer_state["q_proj"], layer_state["k_proj"], layer_state["v_proj"]], 
                        block=64, n_iter=4)
    layer_state["input_ln"].mul_(t_qkv.to(dtype))
    layer_state["q_proj"] = torch.matmul(layer_state["q_proj"], torch.diag(1 / t_qkv)).to(dtype)
    layer_state["k_proj"] = torch.matmul(layer_state["k_proj"], torch.diag(1 / t_qkv)).to(dtype)
    layer_state["v_proj"] = torch.matmul(layer_state["v_proj"], torch.diag(1 / t_qkv)).to(dtype)
    del t_qkv
    gc.collect()
    
    t_gu = sink_scale([layer_state["gate_proj"], layer_state["up_proj"]], block=64, n_iter=4)
    layer_state["post_ln"].mul_(t_gu.to(dtype))
    layer_state["gate_proj"] = torch.matmul(layer_state["gate_proj"], torch.diag(1 / t_gu)).to(dtype)
    layer_state["up_proj"] = torch.matmul(layer_state["up_proj"], torch.diag(1 / t_gu)).to(dtype)
    del t_gu
    gc.collect()
    
    t_d = sink_scale([layer_state["down_proj"]], block=64, n_iter=4)
    layer_state["down_proj"] = torch.matmul(layer_state["down_proj"], torch.diag(1 / t_d)).to(dtype)
    layer_state["up_proj"] = torch.matmul(torch.diag(t_d), layer_state["up_proj"]).to(dtype)
    del t_d
    gc.collect()
    
    n_group = layer_state["v_proj"].shape[0] // layer_state["o_proj"].shape[0]
    if n_group > 0:
        oOut, _ = layer_state["o_proj"].shape
        o_weight = layer_state["o_proj"].reshape(n_group * oOut, -1)
        t_o = sink_scale([o_weight], block=64, n_iter=4)
        del o_weight
        gc.collect()
        
        layer_state["v_proj"] = torch.matmul(torch.diag(t_o), layer_state["v_proj"]).to(dtype)
        t_o_cat = torch.cat([t_o] * n_group)
        layer_state["o_proj"] = torch.matmul(layer_state["o_proj"], torch.diag(1 / t_o_cat)).to(dtype)
        del t_o, t_o_cat
        gc.collect()


def layer_keys(layer_idx):
    return {
        "q_proj": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
        "k_proj": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
        "v_proj": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        "o_proj": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        "gate_proj": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
        "up_proj": f"model.layers.{layer_idx}.mlp.up_proj.weight",
        "down_proj": f"model.layers.{layer_idx}.mlp.down_proj.weight",
        "input_ln": f"model.layers.{layer_idx}.input_layernorm.weight",
        "post_ln": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
    }


def load_model_shards(model_dir: Path):
    shards = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        from safetensors.torch import load_file
        data = load_file(sf, device="cpu")
        for key in data:
            if "model.layers" in key:
                idx = int(key.split("model.layers.")[1].split(".")[0])
                if idx not in shards:
                    shards[idx] = {}
                shards[idx][key] = data[key]
        del data
        gc.collect()
    
    num_layers = max(shards.keys()) + 1 if shards else 0
    
    embed = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        from safetensors.torch import load_file
        data = load_file(sf, device="cpu")
        for k in list(data.keys()):
            if "model.layers" not in k and "lm_head" not in k and "model.norm" not in k:
                embed[k] = data.pop(k)
        if data:
            pass
        del data
        gc.collect()
        if embed:
            break
    
    lm_head = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        from safetensors.torch import load_file
        data = load_file(sf, device="cpu")
        for k in list(data.keys()):
            if "lm_head" in k:
                lm_head[k] = data.pop(k)
        del data
        gc.collect()
        if lm_head:
            break
    
    model_norm = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        from safetensors.torch import load_file
        data = load_file(sf, device="cpu")
        for k in list(data.keys()):
            if "model.norm" in k:
                model_norm[k] = data.pop(k)
        del data
        gc.collect()
        if model_norm:
            break
    
    return shards, num_layers, embed, lm_head, model_norm


def save_model(output_dir: Path, shards, num_layers, embed, lm_head, model_norm):
    from safetensors.torch import save_file
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = {}
    all_data.update(embed)
    all_data.update(model_norm)
    all_data.update(lm_head)
    
    layer_keys_all = set()
    for layer_data in shards.values():
        layer_keys_all.update(layer_data.keys())
    
    for key in sorted(layer_keys_all):
        all_data[key] = shards[key // 1000][key] if key in shards[key // 1000] else None
    
    for layer_idx in range(num_layers):
        for prefix, full_key in layer_keys(layer_idx).items():
            for sf in sorted(output_dir.parent.glob("*.safetensors")):
                from safetensors.torch import load_file
                try:
                    data = load_file(sf, device="cpu")
                    if full_key in data:
                        all_data[full_key] = data[full_key]
                        break
                    del data
                except:
                    pass
    
    for key in list(all_data.keys()):
        if all_data[key] is None:
            del all_data[key]
    
    shard_size = 5 * 1024 * 1024 * 1024
    current_shard = {}
    current_size = 0
    shard_idx = 0
    
    for key, tensor in sorted(all_data.items()):
        tensor_size = tensor.element_size() * tensor.nelement()
        if current_size + tensor_size > shard_size and current_shard:
            save_file(current_shard, output_dir / f"model-{shard_idx:05d}-of-00001.safetensors")
            del current_shard
            gc.collect()
            shard_idx += 1
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size
    
    if current_shard:
        save_file(current_shard, output_dir / f"model-{shard_idx:05d}-of-00001.safetensors")
        del current_shard
        gc.collect()
    
    return output_dir


def process_model_full(model_dir: str, output_dir: str, group_size=64, n_iter=4, n_repeat=3):
    model_path = Path(model_dir)
    out_path = Path(output_dir)
    
    print("Loading model structure...")
    shards, num_layers, embed, lm_head, model_norm = load_model_shards(model_path)
    print(f"Found {num_layers} layers")
    print(f"Peak RAM so far: {get_ram_usage():.1f} MB")
    
    for repeat in range(n_repeat):
        print(f"\n=== Repeat {repeat+1}/{n_repeat} ===")
        for layer_idx in tqdm(range(num_layers), desc=f"Rep {repeat+1}"):
            if layer_idx not in shards:
                continue
            
            state = {
                "q_proj": shards[layer_idx].get(layer_keys(layer_idx)["q_proj"]),
                "k_proj": shards[layer_idx].get(layer_keys(layer_idx)["k_proj"]),
                "v_proj": shards[layer_idx].get(layer_keys(layer_idx)["v_proj"]),
                "o_proj": shards[layer_idx].get(layer_keys(layer_idx)["o_proj"]),
                "gate_proj": shards[layer_idx].get(layer_keys(layer_idx)["gate_proj"]),
                "up_proj": shards[layer_idx].get(layer_keys(layer_idx)["up_proj"]),
                "down_proj": shards[layer_idx].get(layer_keys(layer_idx)["down_proj"]),
                "input_ln": shards[layer_idx].get(layer_keys(layer_idx)["input_ln"]),
                "post_ln": shards[layer_idx].get(layer_keys(layer_idx)["post_ln"]),
            }
            
            if state["q_proj"] is None:
                continue
            
            apply_presinq_layer(state)
            
            for k, v in state.items():
                full_key = layer_keys(layer_idx)[k]
                shards[layer_idx][full_key] = v
            
            del state
            gc.collect()
        
        print(f"After repeat {repeat+1}: Peak RAM {get_ram_usage():.1f} MB")
        
        if repeat < n_repeat - 1:
            ckpt = out_path.parent / f"{out_path.name}_rep{repeat+1}"
            print(f"Saving checkpoint to {ckpt}...")
            save_model(ckpt, shards, num_layers, embed, lm_head, model_norm)
    
    print(f"\nSaving final model to {out_path}...")
    save_model(out_path, shards, num_layers, embed, lm_head, model_norm)


def get_ram_usage():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    total = 3409212
                    avail = int(line.split()[1]) * 1024
                    return (total - avail) / (1024 * 1024)
    except:
        pass
    return 0


def main():
    parser = argparse.ArgumentParser(description="Memory-optimized Pre-SINQ")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default="./qwen25-0.5b-presinq")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--n_iter", type=int, default=4)
    parser.add_argument("--n_repeat", type=int, default=3)
    parser.add_argument("--download", action="store_true",
                        help="Download model from HuggingFace")
    args = parser.parse_args()
    
    if args.download:
        print(f"Downloading {args.model_name}...")
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        
        cache = Path(snapshot_download(args.model_name))
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model_dir = cache
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(out_dir)
        
        import shutil
        for f in cache.glob("*.json"):
            shutil.copy(f, out_dir / f.name)
        for f in cache.glob("tokenizer*"):
            if f.is_file():
                shutil.copy(f, out_dir / f.name)
        
        model_dir = str(cache)
    else:
        model_dir = args.model_name
    
    process_model_full(
        model_dir, args.output_dir,
        group_size=args.group_size,
        n_iter=args.n_iter,
        n_repeat=args.n_repeat
    )
    
    print(f"\nDone! To convert to GGUF:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {args.output_dir} --outfile model-f16.gguf --outtype f16")


if __name__ == "__main__":
    main()
