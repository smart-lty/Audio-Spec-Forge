#!/bin/bash
set -x

cd /cpfs04/user/lvqidan/SpecForge

source ~/anaconda3/bin/activate specforge

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# Train Qwen2-audio Eagle3 for multimodal inference acceleration
NUM_GPUS=${1:-2}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_qwen2_audio_eagle3_online.py \
    --target-model-path /cpfs04/user/lvqidan/speechllm/checkpoints/Qwen2-Audio-7B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen2-audio-7B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/multimodal_sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen2-audio-7b-eagle3 \
    --num-epochs 5 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --ttt-length 7 \
    --chat-template qwen2_audio \
    --cache-dir $ROOT_DIR/cache \
    --bf16 \
    --save-steps 500 \
    --log-steps 10