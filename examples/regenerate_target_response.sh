#!/bin/bash

# Example script for Eagle3 Step 1: Target Response Generation
# This script generates target model responses for the audio dataset

# Configuration
TARGET_MODEL_PATH="/cpfs04/user/lvqidan/speechllm/checkpoints/Qwen2-Audio-7B-Instruct"
INPUT_DATA="/cpfs04/user/lvqidan/SpecForge/cache/dataset/gigaspeech_demo.jsonl"
OUTPUT_DATA="/cpfs04/user/lvqidan/SpecForge/cache/dataset/gigaspeech_demo_regenerate.jsonl"

# Generation parameters
MAX_LENGTH=2048
TEMPERATURE=0
TOP_P=0.9

# System parameters
DEVICE="auto"
TORCH_DTYPE="bfloat16"


python scripts/generate_target_responses.py \
    --input-data "$INPUT_DATA" \
    --output-data "$OUTPUT_DATA" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --max-length $MAX_LENGTH \
    --temperature $TEMPERATURE \
    --device "$DEVICE" \
    --torch-dtype "$TORCH_DTYPE" \
    --resume