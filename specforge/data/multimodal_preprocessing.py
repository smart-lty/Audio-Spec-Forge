import json
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from specforge.utils import padding
from .template import TEMPLATE_REGISTRY, ChatTemplate

# define a type called multimodal conversation
MultimodalConversation = List[Dict[str, Union[str, List[Dict[str, str]]]]]


def load_audio_file(
    audio_path: str,
    target_sample_rate: int = 16000,
    max_duration: Optional[float] = None
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sampling rate for audio
        max_duration: Maximum duration in seconds (None for no limit)
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    try:
        # Load with duration limit if specified
        duration = max_duration if max_duration is not None else None
        audio, sr = librosa.load(
            audio_path, 
            sr=target_sample_rate, 
            duration=duration,
            mono=True,
            dtype=np.float32
        )
        
        # Ensure audio is 1D and not empty
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        if len(audio) == 0:
            # Create minimal silence if audio is empty
            audio = np.zeros(int(target_sample_rate * 0.1), dtype=np.float32)
            
        return torch.tensor(audio, dtype=torch.float32), sr
    except Exception as e:
        print(f"⚠️  Audio loading failed for {audio_path}: {e}")
        # Create fallback audio (1 second of silence)
        fallback_audio = np.zeros(target_sample_rate, dtype=np.float32)
        return torch.tensor(fallback_audio, dtype=torch.float32), target_sample_rate


def preprocess_multimodal_conversations(
    processor,  # AutoProcessor for Qwen2Audio
    conversations: List[MultimodalConversation],
    wav_paths: List[str],
    ground_truths: List[str],
    max_length: int = 2048,
    max_audio_duration: Optional[float] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess a batch of multimodal conversations with audio for Eagle3 Step 2 training.
    
    This function expects that Step 1 (target model response generation) has already been completed,
    and assistant responses in conversations contain the pre-generated target model outputs.

    Args:
        processor: The Qwen2Audio processor to use for processing.
        conversations: A list of conversations with pre-generated assistant responses from Step 1.
        wav_paths: List of paths to audio files.
        ground_truths: List of ground truth transcriptions (not used in Step 2 training).
        max_length: The maximum length of the tokenized input.
        max_audio_duration: Maximum audio duration in seconds.

    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - attention_mask: List of attention masks.
            - loss_mask: List of loss masks indicating which tokens should contribute to the loss.
            - input_features: List of audio feature tensors.
            - feature_attention_mask: List of audio attention masks.
    """
    # Prepare result
    results = {
        "input_ids": [],
        "attention_mask": [],
        "loss_mask": [],
        "input_features": [],
        "feature_attention_mask": []
    }

    for conversation, wav_path, ground_truth in zip(conversations, wav_paths, ground_truths):
        if not conversation:
            # Skip empty conversations
            continue
            
        # Transform to multimodal format
        multimodal_conversation = []
        
        for message in conversation:
            if message["role"] == "user":
                # Add audio and text content for user message
                multimodal_conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": wav_path},
                        {"type": "text", "text": message["content"]}
                    ]
                })
            elif message["role"] == "assistant":
                # Use pre-generated assistant response from Step 1
                # Assert that content is not empty (should be pre-generated)
                assert message["content"], f"Assistant content is empty for {wav_path}. Please run Step 1 first to generate target model responses."
                multimodal_conversation.append({
                    "role": "assistant", 
                    "content": message["content"]
                })
        
        try:
            # Apply chat template
            text = processor.apply_chat_template(
                multimodal_conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Load audio
            audio, sample_rate = load_audio_file(
                wav_path, 
                target_sample_rate=processor.feature_extractor.sampling_rate,
                max_duration=max_audio_duration
            )
            
            # Process with Qwen2Audio processor
            # Convert audio tensor to numpy for processor
            if isinstance(audio, torch.Tensor):
                audio_array = audio.numpy()
            else:
                audio_array = audio
                
            inputs = processor(
                text=text, 
                audios=[audio_array], 
                return_tensors="pt", 
                padding=True,
                max_length=max_length,
                truncation=True,
                sampling_rate=processor.feature_extractor.sampling_rate,  # Fix sampling rate warning
            )
            
            # Extract tensors and squeeze batch dimension
            input_ids = inputs.input_ids.squeeze(0)
            attention_mask = inputs.attention_mask.squeeze(0)
            input_features = inputs.input_features.squeeze(0)
            feature_attention_mask = inputs.feature_attention_mask.squeeze(0)
            
            # Create loss mask - simplified for Step 2 (will be handled by Eagle3 training)
            loss_mask = torch.ones(len(input_ids), dtype=torch.long)  # Simple: all tokens for loss
            
            # Add to results
            results["input_ids"].append(input_ids[None, :])
            results["attention_mask"].append(attention_mask[None, :])
            results["loss_mask"].append(loss_mask[None, :])
            results["input_features"].append(input_features[None, :])
            results["feature_attention_mask"].append(feature_attention_mask[None, :])
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue
            
    return results


class AudioDataset(Dataset):
    """
    Custom dataset class for loading audio data from JSONL files for Eagle3 Step 2 training.
    
    Expected JSONL format after Step 1 (target model response generation):
    {
        "id": "unique_id",
        "wav_path": "/path/to/audio.wav", 
        "conversations": [
            {"role": "user", "content": "..."}, 
            {"role": "assistant", "content": "pre-generated target model response"}
        ],
        "Ground_Truth": "original ground truth (not used in Step 2 training)"
    }
    
    Note: This assumes Step 1 has been completed and assistant responses are pre-generated.
    """
    
    def __init__(
        self,
        data_path: str,
        processor, 
        max_length: int = 2048,
        max_audio_duration: Optional[float] = None,
        transform=None,
    ):
        """
        Initialize AudioDataset.
        
        Args:
            data_path: Path to JSONL file containing audio data
            processor: AutoProcessor for Qwen2Audio
            max_length: Maximum sequence length
            max_audio_duration: Maximum audio duration in seconds
            transform: Optional transform to apply to data
        """
        self.data_path = data_path
        self.processor = processor
        self.max_length = max_length
        self.max_audio_duration = max_audio_duration
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    # Validate required fields
                    if all(key in item for key in ['wav_path', 'conversations', 'Ground_Truth']):
                        # Check if audio file exists
                        if os.path.exists(item['wav_path']):
                            data.append(item)
                        else:
                            print(f"Warning: Audio file not found: {item['wav_path']}")
                    else:
                        print(f"Warning: Missing required fields in line {line_num}")
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in line {line_num}: {e}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.data[idx]
        
        try:
            # Process single item
            results = preprocess_multimodal_conversations(
                processor=self.processor,
                conversations=[item['conversations']],
                wav_paths=[item['wav_path']],
                ground_truths=[item['Ground_Truth']],
                max_length=self.max_length,
                max_audio_duration=self.max_audio_duration,
            )
            
            # Extract first (and only) item from each list
            processed_item = {}
            for key in results:
                if results[key]:  # Check if list is not empty
                    processed_item[key] = results[key][0].squeeze(0)  # Remove batch dimension
                else:
                    # Handle empty results - create dummy tensors
                    if key in ['input_ids', 'attention_mask', 'loss_mask']:
                        processed_item[key] = torch.tensor([0], dtype=torch.long)
                    else:  # audio features
                        processed_item[key] = torch.tensor([[0]], dtype=torch.float32)
            
            # Add attention mask for consistency
            if 'attention_mask' not in processed_item:
                processed_item['attention_mask'] = torch.ones_like(processed_item['input_ids'])
            
            # Apply transform if provided
            if self.transform:
                processed_item = self.transform(processed_item)
                
            return processed_item
            
        except Exception as e:
            print(f"Error processing item {idx} ({item.get('id', 'unknown')}): {e}")
            # Return dummy item to prevent dataset failure
            return {
                'input_ids': torch.tensor([0], dtype=torch.long),
                'attention_mask': torch.tensor([1], dtype=torch.long),
                'loss_mask': torch.tensor([0], dtype=torch.long),
                'input_features': torch.tensor([[0]], dtype=torch.float32),
                'feature_attention_mask': torch.tensor([[1]], dtype=torch.float32),
            }


def build_multimodal_eagle3_dataset(
    data_path: str,
    processor, 
    max_length: Optional[int] = 2048,
    max_audio_duration: Optional[float] = None,
    transform=None,
) -> AudioDataset:
    """
    Build multimodal eagle3 dataset for Eagle3 Step 2 training (audio + text processing).
    
    This function is designed for Step 2 of Eagle3 training, assuming Step 1 (target model
    response generation) has already been completed and the dataset contains pre-generated
    assistant responses.
    
    Args:
        data_path: Path to JSONL file containing audio data with pre-generated responses
        processor: AutoProcessor for Qwen2Audio (handles both text and audio)
        max_length: Maximum length of tokenized input
        max_audio_duration: Maximum audio duration in seconds
        transform: Optional transform to apply to data
        
    Returns:
        AudioDataset instance ready for Eagle3 Step 2 training
    """
    return AudioDataset(
        data_path=data_path,
        processor=processor,
        max_length=max_length,
        max_audio_duration=max_audio_duration,
        transform=transform,
    )


# For backward compatibility - collate function for DataLoader
def multimodal_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for multimodal data that handles variable-length sequences.
    
    Args:
        batch: List of processed items from AudioDataset
        
    Returns:
        Batched tensors with proper padding
    """
    # Separate keys for different tensor types
    text_keys = ['input_ids', 'attention_mask', 'loss_mask']
    audio_keys = ['input_features', 'feature_attention_mask']
    
    result = {}
    
    # Handle text sequences with padding
    for key in text_keys:
        if key in batch[0]:
            sequences = [item[key] for item in batch]
            # First pad each sequence to same length, then stack
            max_len = max(seq.shape[0] for seq in sequences)
            padded_sequences = []
            for seq in sequences:
                # Pad sequence to max_len
                if seq.shape[0] < max_len:
                    pad_size = max_len - seq.shape[0]
                    if key in ['input_ids', 'attention_mask', 'loss_mask']:
                        # Pad with 0 for these keys
                        padded = torch.cat([seq, torch.zeros(pad_size, dtype=seq.dtype)])
                    else:
                        padded = torch.cat([seq, torch.zeros(pad_size, dtype=seq.dtype)])
                else:
                    padded = seq
                padded_sequences.append(padded)
            
            result[key] = torch.stack(padded_sequences)
    
    # Handle audio features
    for key in audio_keys:
        if key in batch[0]:
            # Stack audio features directly (assuming they have consistent shape from processor)
            features = [item[key] for item in batch]
            result[key] = torch.stack(features)
    
    return result