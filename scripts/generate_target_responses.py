#!/usr/bin/env python3
"""
Step 1: Generate target model responses for Eagle3 multimodal training.

This script processes the original dataset and generates responses using the target model,
preparing the data for Step 2 (Eagle3 draft model training).

Input: JSONL with empty assistant responses
Output: JSONL with target model generated assistant responses
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor

# Suppress transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from specforge.modeling.target.qwen2_audio import Qwen2AudioForConditionalGeneration


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate target model responses for Eagle3 Step 1"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--input-data", 
        type=str, 
        required=True,
        help="Path to input JSONL file with original conversations"
    )
    parser.add_argument(
        "--output-data", 
        type=str, 
        required=True,
        help="Path to output JSONL file with generated responses"
    )
    
    # Model configuration
    parser.add_argument(
        "--target-model-path", 
        type=str, 
        required=True,
        help="Path to target model (Qwen2-Audio)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=2048,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--max-audio-duration", 
        type=float, 
        default=None,
        help="Maximum audio duration in seconds"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size for generation (currently only supports 1)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Generation temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=1.0,
        help="Top-p sampling parameter (not used with temperature=0)"
    )
    parser.add_argument(
        "--do-sample", 
        action="store_true",
        help="Use sampling for generation (default: greedy decoding)"
    )
    
    # System configuration
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use for generation (auto/cuda/cpu)"
    )
    parser.add_argument(
        "--torch-dtype", 
        type=str, 
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model"
    )
    
    # Processing options
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume from existing output file"
    )
    parser.add_argument(
        "--skip-existing", 
        action="store_true",
        help="Skip items that already have assistant responses"
    )
    
    return parser.parse_args()


def load_audio_file(
    audio_path: str,
    target_sample_rate: int = 16000,
    max_duration: Optional[float] = None
) -> torch.Tensor:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sampling rate
        max_duration: Maximum duration in seconds
        
    Returns:
        Audio tensor
    """
    try:
        # Load audio with proper parameters
        duration = max_duration if max_duration is not None else None
        audio, sr = librosa.load(
            audio_path, 
            sr=target_sample_rate, 
            duration=duration,
            mono=True,  # Ensure mono audio
            dtype=np.float32
        )
        
        # Ensure audio is 1D and not empty
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        if len(audio) == 0:
            # Create minimal silence if audio is empty
            audio = np.zeros(int(target_sample_rate * 0.1), dtype=np.float32)  # 0.1 second silence
        
        return torch.tensor(audio, dtype=torch.float32)
    except Exception as e:
        # Create fallback audio (1 second of silence)
        print(f"⚠️  Audio loading failed for {audio_path}: {e}")
        print(f"   Using fallback silence audio")
        fallback_audio = np.zeros(target_sample_rate, dtype=np.float32)
        return torch.tensor(fallback_audio, dtype=torch.float32)


def prepare_multimodal_input(
    processor,
    conversation: List[Dict[str, str]],
    wav_path: str,
    max_audio_duration: Optional[float] = None
) -> Dict[str, torch.Tensor]:
    """
    Prepare multimodal input for target model generation.
    
    Args:
        processor: AutoProcessor for Qwen2Audio
        conversation: Original conversation (with empty assistant response)
        wav_path: Path to audio file
        max_audio_duration: Maximum audio duration
        
    Returns:
        Processed inputs for target model
    """
    # Create multimodal conversation for generation (user message only)
    multimodal_conversation = []
    
    for message in conversation:
        if message["role"] == "user":
            multimodal_conversation.append({
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": wav_path},
                    {"type": "text", "text": message["content"]}
                ]
            })
        # Skip assistant messages - we want to generate them
    
    # Apply chat template (add generation prompt)
    text = processor.apply_chat_template(
        multimodal_conversation, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Load and process audio
    audio = load_audio_file(
        wav_path, 
        target_sample_rate=processor.feature_extractor.sampling_rate,
        max_duration=max_audio_duration
    )
    
    # Process with Qwen2Audio processor
    try:
        # Ensure audio is numpy array for processor
        if isinstance(audio, torch.Tensor):
            audio_array = audio.numpy()
        else:
            audio_array = audio
            
        inputs = processor(
            text=text, 
            audios=[audio_array], 
            return_tensors="pt", 
            padding=True,
            sampling_rate=processor.feature_extractor.sampling_rate,  # Explicitly set sampling rate
        )
    except Exception as e:
        print(f"⚠️  Processor error for {wav_path}: {e}")
        # Try with minimal silence audio as fallback
        fallback_audio = np.zeros(processor.feature_extractor.sampling_rate, dtype=np.float32)
        inputs = processor(
            text=text, 
            audios=[fallback_audio], 
            return_tensors="pt", 
            padding=True,
            sampling_rate=processor.feature_extractor.sampling_rate,
        )
    
    return inputs


def generate_response(
    model,
    processor,
    inputs: Dict[str, torch.Tensor],
    max_length: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
) -> str:
    """
    Generate response using target model.
    
    Args:
        model: Target model
        processor: AutoProcessor
        inputs: Processed inputs
        max_length: Maximum generation length
        temperature: Generation temperature
        top_p: Top-p sampling
        do_sample: Whether to use sampling
        
    Returns:
        Generated response text
    """
    with torch.no_grad():
        # Move inputs to model device
        device_inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        generation_kwargs = {
            "max_length": max_length,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        # Only add sampling parameters if do_sample is True
        if do_sample:
            generation_kwargs["do_sample"] = True
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
        else:
            # Explicitly set do_sample=False for greedy decoding
            generation_kwargs["do_sample"] = False
        
        outputs = model.generate(**device_inputs, **generation_kwargs)
        
        # Decode response (skip input tokens)
        input_length = device_inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_length:]
        response = processor.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return response.strip()


def load_existing_data(output_path: str) -> Dict[str, Dict]:
    """
    Load existing output data for resuming.
    
    Args:
        output_path: Path to existing output file
        
    Returns:
        Dictionary mapping item IDs to processed items
    """
    existing_data = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    existing_data[item['id']] = item
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing_data


def main():
    """Main function."""
    args = parse_args()
    
    # Setup torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16, 
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    print("=== Eagle3 Step 1: Target Response Generation ===")
    print(f"Input: {args.input_data}")
    print(f"Output: {args.output_data}")
    print(f"Target Model: {args.target_model_path}")
    print(f"Device: {args.device}")
    print(f"Torch dtype: {args.torch_dtype}")
    
    # Load target model and processor
    print("\n1. Loading target model and processor...")
    processor = AutoProcessor.from_pretrained(args.target_model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.target_model_path,
        torch_dtype=torch_dtype,
        device_map=args.device
    ).eval()
    print(f"✅ Model loaded on device: {model.device}")
    
    # Load input data
    print(f"\n2. Loading input data from {args.input_data}...")
    input_data = []
    with open(args.input_data, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                # Validate required fields
                required_fields = ['id', 'wav_path', 'conversations']
                if all(field in item for field in required_fields):
                    input_data.append(item)
                else:
                    print(f"⚠️  Skipping line {line_num}: Missing required fields")
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping line {line_num}: Invalid JSON - {e}")
    
    print(f"✅ Loaded {len(input_data)} items")
    
    # Load existing data if resuming
    existing_data = {}
    processed_count = 0
    if args.resume:
        print(f"\n3. Loading existing data for resume...")
        existing_data = load_existing_data(args.output_data)
        processed_count = len(existing_data)
        print(f"✅ Found {processed_count} existing items")
    
    # Prepare output file
    os.makedirs(os.path.dirname(args.output_data), exist_ok=True)
    
    # Process data
    print(f"\n4. Generating responses...")
    total_items = len(input_data)
    skipped_count = 0
    error_count = 0
    
    with open(args.output_data, 'w' if not args.resume else 'a', encoding='utf-8') as out_f:
        # If not resuming, write existing data first
        if not args.resume:
            for existing_item in existing_data.values():
                out_f.write(json.dumps(existing_item, ensure_ascii=False) + '\n')
        
        progress_bar = tqdm(input_data, desc="Processing", unit="item")
        
        for item in progress_bar:
            item_id = item['id']
            
            # Skip if already processed
            if item_id in existing_data:
                skipped_count += 1
                continue
            
            # Skip if already has assistant response and skip_existing is True
            if args.skip_existing:
                conversations = item.get('conversations', [])
                has_assistant_response = any(
                    msg.get('role') == 'assistant' and msg.get('content', '').strip()
                    for msg in conversations
                )
                if has_assistant_response:
                    skipped_count += 1
                    # Still write to output
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    out_f.flush()
                    continue
            
            try:
                # Check if audio file exists
                wav_path = item['wav_path']
                if not os.path.exists(wav_path):
                    print(f"⚠️  Audio file not found: {wav_path}")
                    error_count += 1
                    continue
                
                # Prepare input
                inputs = prepare_multimodal_input(
                    processor=processor,
                    conversation=item['conversations'],
                    wav_path=wav_path,
                    max_audio_duration=args.max_audio_duration
                )
                
                # Generate response
                response = generate_response(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                )
                
                # Update conversations with generated response
                updated_conversations = []
                for msg in item['conversations']:
                    if msg['role'] == 'user':
                        updated_conversations.append(msg)
                    elif msg['role'] == 'assistant':
                        updated_conversations.append({
                            'role': 'assistant',
                            'content': response
                        })
                
                # Create updated item
                updated_item = item.copy()
                updated_item['conversations'] = updated_conversations
                
                # Write to output
                out_f.write(json.dumps(updated_item, ensure_ascii=False) + '\n')
                out_f.flush()
                
                processed_count += 1
                progress_bar.set_postfix({
                    'processed': processed_count,
                    'errors': error_count,
                    'skipped': skipped_count
                })
                
            except Exception as e:
                print(f"❌ Error processing {item_id}: {e}")
                error_count += 1
                continue
    
    print(f"\n=== Step 1 Complete ===")
    print(f"📊 Total items: {total_items}")
    print(f"✅ Processed: {processed_count}")  
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📁 Output saved to: {args.output_data}")
    
    if error_count == 0:
        print("🎉 All items processed successfully! Ready for Step 2.")
    else:
        print("⚠️  Some items had errors. Please check the logs.")


if __name__ == "__main__":
    main()