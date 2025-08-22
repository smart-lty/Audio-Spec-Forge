#!/bin/bash
set -x

NUM_GPUS=${1:-2}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    -m \
    scripts.train_audio_eagle3_online \
    --target-model-path /cpfs04/user/lvqidan/speechllm/checkpoints/Qwen2-Audio-7B-Instruct \
    --draft-model-config configs/qwen2-audio-eagle3.json \
    --train-data-path cache/dataset/gigaspeech_demo_regenerate.jsonl \
    --output-dir outputs/qwen2-audio-7b-eagle3 \
    --num-epochs 10 \
    --batch-size 5 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --ttt-length 7 \
    --cache-dir cache 