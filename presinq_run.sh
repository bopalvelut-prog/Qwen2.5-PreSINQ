#!/bin/bash
# PreSINQ Optimized Runner for Low-Spec CPU (2 threads, Celeron)
# Usage: bash presinq_run.sh [model] [output_dir]

set -e

MODEL="${1:-Qwen/Qwen2.5-0.5B}"
OUTPUT="${2:-./qwen25-0.5b-presinq}"
GROUP_SIZE=64
N_ITER=2
N_REPEAT=1

# CPU optimization
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "============================================================"
echo "PreSINQ Runner (Optimized for Low-Spec CPU)"
echo "============================================================"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT"
echo "  CPU threads: $(nproc)"
echo "  RAM: $(free -h | awk '/Mem:/{print $2}')"
echo "  Config: group_size=$GROUP_SIZE, n_iter=$N_ITER, n_repeat=$N_REPEAT"
echo "============================================================"

# Run PreSINQ
python3 /home/m/Qwen2.5-PreSINQ/presinq_qwen25_cpu.py \
  --model_name "$MODEL" \
  --output_dir "$OUTPUT" \
  --group_size "$GROUP_SIZE" \
  --n_iter "$N_ITER" \
  --n_repeat "$N_REPEAT" \
  --skip_gguf

echo ""
echo "Done! PreSINQ model saved to: $OUTPUT"
echo ""
echo "To convert to GGUF (on a machine with more RAM):"
echo "  python llama.cpp/convert_hf_to_gguf.py $OUTPUT --outfile model-f16.gguf --outtype f16"
echo "  ./llama.cpp/llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M"
