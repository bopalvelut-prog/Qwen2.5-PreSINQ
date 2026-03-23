# Qwen2.5-0.5B PreSINQ

CPU-compatible Pre-SINQ (Sinkhorn Normalization) for Qwen2.5-0.5B, based on [huawei-csl/SINQ](https://github.com/huawei-csl/SINQ).

## What is Pre-SINQ?

Pre-SINQ applies Sinkhorn-inspired weight reparameterization to make model weights easier to quantize. The model output is mathematically identical to the original - no accuracy loss.

## Quick Start

```bash
# Install dependencies
pip install torch transformers tqdm

# Run Pre-SINQ (CPU mode)
python presinq_qwen25_cpu.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --output_dir ./qwen25-0.5b-presinq \
  --n_repeat 2 \
  --n_iter 4
```

## Convert to GGUF

```bash
# Convert to GGUF FP16
python llama.cpp/convert_hf_to_gguf.py ./qwen25-0.5b-presinq \
  --outfile qwen25-0.5b-presinq-f16.gguf --outtype f16

# Quantize to Q4_K_M
./llama.cpp/llama-quantize qwen25-0.5b-presinq-f16.gguf \
  qwen25-0.5b-presinq-Q4_K_M.gguf Q4_K_M
```

## HuggingFace Model

Pre-quantized GGUF: [bopalvelut-prog/Qwen2.5-0.5B-PreSINQ-GGUF](https://huggingface.co/bopalvelut-prog/Qwen2.5-0.5B-PreSINQ-GGUF)

## Based On

- [huawei-csl/SINQ](https://github.com/huawei-csl/SINQ) - Original SINQ implementation
- [bopalvelut-prog/prima.cpp](https://github.com/bopalvelut-prog/prima.cpp) - Inference engine

## License

Apache 2.0 (same as SINQ)
