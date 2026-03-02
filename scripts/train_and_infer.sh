#!/bin/bash
# Train TransESC with leak-free data, then run inference
set -e

export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="./blenderbot_small-90M"
OUTPUT_DIR="./blender_strategy"
LOG_DIR="./logs"

echo "============================================"
echo "Waiting for current training to finish..."
echo "============================================"

# Wait for existing python training process to finish
while pgrep -f "python.*main.py.*blender_strategy" > /dev/null 2>&1; do
    sleep 30
done

echo "Training process finished."

# Check if checkpoint exists
if [ ! -d "$OUTPUT_DIR" ] || [ -z "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
    echo "ERROR: No checkpoint found in $OUTPUT_DIR"
    exit 1
fi

echo ""
echo "============================================"
echo "Starting inference (test set)..."
echo "============================================"

conda run -n transesc python -u main.py \
    --model_name_or_path "$MODEL_PATH" \
    --config_name "$MODEL_PATH" \
    --tokenizer_name "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --test 2>&1 | tee "$LOG_DIR/infer_leakfree_70_15_15.log"

echo ""
echo "============================================"
echo "Inference complete!"
echo "Results in: ./generated_data/"
echo "============================================"
