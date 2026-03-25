#!/bin/bash
# Quantize Qwen2.5-0.5B-PreSINQ to all quantization formats
set -e

MODEL_NAME="Qwen/Qwen2.5-0.5B"
PRESINQ_DIR="./qwen25-0.5b-presinq"
GGUF_DIR="./gguf"
PRIMA_BUILD="/home/m/prima.cpp/build"
QUANTIZE="$PRIMA_BUILD/bin/llama-quantize"
CONVERT="/home/m/prima.cpp/convert_hf_to_gguf.py"

# Step 1: Run PreSINQ
if [ ! -d "$PRESINQ_DIR" ]; then
    echo "=== Step 1: Running PreSINQ ==="
    PYTHONUNBUFFERED=1 python3 /home/m/Qwen2.5-PreSINQ/presinq_qwen25_cpu.py \
        --model_name "$MODEL_NAME" \
        --output_dir "$PRESINQ_DIR" \
        --n_repeat 1 --n_iter 2 --skip_gguf
else
    echo "=== PreSINQ already done ==="
fi

# Step 2: Convert to F16 GGUF
mkdir -p "$GGUF_DIR"
F16="$GGUF_DIR/qwen25-0.5b-presinq-f16.gguf"
if [ ! -f "$F16" ]; then
    echo "=== Step 2: Converting to F16 GGUF ==="
    python3 "$CONVERT" "$PRESINQ_DIR" --outfile "$F16" --outtype f16
else
    echo "=== F16 GGUF already exists ==="
fi

# Step 3: Quantize to all formats
echo "=== Step 3: Quantizing ==="
FORMATS=(
    Q4_0 Q4_1 Q5_0 Q5_1
    Q2_K Q2_K_S
    Q3_K_S Q3_K_M Q3_K_L
    Q4_K_S Q4_K_M
    Q5_K_S Q5_K_M
    Q6_K
    Q8_0
    IQ2_XXS IQ2_XS IQ2_S IQ2_M
    IQ3_XXS IQ3_S IQ3_M IQ3_XS
    IQ4_NL IQ4_XS
    TQ1_0 TQ2_0
    IQ1_S IQ1_M
)

for fmt in "${FORMATS[@]}"; do
    OUT="$GGUF_DIR/qwen25-0.5b-presinq-${fmt,,}.gguf"
    if [ -f "$OUT" ]; then
        echo "  SKIP $fmt (exists)"
        continue
    fi
    echo "  Quantizing $fmt..."
    "$QUANTIZE" "$F16" "$OUT" "$fmt" 2>/dev/null
    SIZE=$(ls -lh "$OUT" | awk '{print $5}')
    echo "  -> $fmt: $SIZE"
done

# Summary
echo ""
echo "============================================================"
echo "All quantizations complete!"
echo "============================================================"
echo "Format      Size"
echo "--------    ------"
for fmt in "${FORMATS[@]}"; do
    OUT="$GGUF_DIR/qwen25-0.5b-presinq-${fmt,,}.gguf"
    if [ -f "$OUT" ]; then
        SIZE=$(ls -lh "$OUT" | awk '{print $5}')
        printf "%-12s %s\n" "$fmt" "$SIZE"
    fi
done
echo "============================================================"
echo "Files in: $GGUF_DIR/"
