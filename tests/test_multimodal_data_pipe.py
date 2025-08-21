import hashlib
import json
import os
import pickle
import time

import psutil
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from specforge.data import (
    build_multimodal_eagle3_dataset,
    multimodal_collate_fn,
)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_cache_key(data_path, processor_path, max_length, max_audio_duration, batch_size):
    """Generate cache key based on configuration"""
    cache_data = {
        "data_path": data_path,
        "processor_path": processor_path,
        "max_length": max_length,
        "max_audio_duration": max_audio_duration,
        "batch_size": batch_size,
    }
    cache_str = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_str.encode()).hexdigest()


def save_dataset_cache(cache_key, dataset, cache_dir):
    """Save dataset cache"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"multimodal_dataset_cache_{cache_key}.pkl")

    # Save dataset object
    cache_data = {
        "dataset": dataset,
        "dataset_length": len(dataset),
    }

    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)

    print(f"Dataset cache saved to: {cache_file}")


def load_dataset_cache(cache_key, cache_dir):
    """Load dataset cache"""
    cache_file = os.path.join(cache_dir, f"multimodal_dataset_cache_{cache_key}.pkl")

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        print(f"Successfully loaded dataset from cache: {cache_file}")
        return cache_data["dataset"]

    except Exception as e:
        print(f"Failed to load dataset cache: {e}")
        return None


def test_dataset_item(dataset, item_idx=0):
    """Test loading a single item from dataset"""
    print(f"\n=== Testing Dataset Item {item_idx} ===")
    try:
        start_time = time.time()
        item = dataset[item_idx]
        load_time = time.time() - start_time
        
        print(f"✅ Item loaded successfully in {load_time:.3f}s")
        print(f"Item keys: {list(item.keys())}")
        
        for key, tensor in item.items():
            print(f"  - {key}: {tensor.shape} ({tensor.dtype})")
            
        # Check for reasonable sizes
        if item["input_ids"].numel() > 1:
            print("✅ input_ids has reasonable size")
        else:
            print("⚠️  input_ids seems too small (likely fallback data)")
            
        if item["input_features"].numel() > 1:
            print("✅ input_features has reasonable size")
        else:
            print("⚠️  input_features seems too small (likely fallback data)")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to load item {item_idx}: {e}")
        return False


def test_dataloader_batch(dataloader, batch_size=2):
    """Test loading a batch from dataloader"""
    print(f"\n=== Testing DataLoader Batch (size={batch_size}) ===")
    try:
        start_time = time.time()
        batch = next(iter(dataloader))
        load_time = time.time() - start_time
        
        print(f"✅ Batch loaded successfully in {load_time:.3f}s")
        print(f"Batch keys: {list(batch.keys())}")
        
        for key, tensor in batch.items():
            print(f"  - {key}: {tensor.shape} ({tensor.dtype})")
            
        # Verify batch size
        expected_batch_size = min(batch_size, len(dataloader.dataset))
        actual_batch_size = batch["input_ids"].shape[0]
        
        if actual_batch_size == expected_batch_size:
            print(f"✅ Correct batch size: {actual_batch_size}")
        else:
            print(f"⚠️  Unexpected batch size: got {actual_batch_size}, expected {expected_batch_size}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=== Multimodal Audio Data Pipeline Test ===")
    print("Testing Eagle3 Step 2 data preparation")
    
    # Configuration
    data_path = "/cpfs04/user/lvqidan/SpecForge/cache/dataset/gigaspeech_demo_regenerate.jsonl"
    processor_path = "/cpfs04/user/lvqidan/speechllm/checkpoints/Qwen2-Audio-7B-Instruct"
    cache_dir = "/cpfs04/user/lvqidan/SpecForge/cache/test_cache"
    
    # Test parameters
    max_length = 512
    max_audio_duration = 5.0
    batch_size = 2
    chat_template = "qwen2_audio"
    
    print(f"\nConfiguration:")
    print(f"  - Data path: {data_path}")
    print(f"  - Processor path: {processor_path}")
    print(f"  - Max length: {max_length}")
    print(f"  - Max audio duration: {max_audio_duration}s")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Chat template: {chat_template}")
    print(f"  - Cache directory: {cache_dir}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please run Step 1 first to generate target responses")
        return
    
    # Record start time and memory
    start_time = time.time()
    start_memory = get_memory_usage()
    print(f"\nStart time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Start memory: {start_memory:.2f} MB")
    
    try:
        # Load processor
        print("\n=== Loading Processor ===")
        processor_start = time.time()
        processor = AutoProcessor.from_pretrained(processor_path)
        processor_time = time.time() - processor_start
        print(f"✅ Processor loaded in {processor_time:.2f}s")
        
        # Generate cache key
        cache_key = get_cache_key(data_path, processor_path, max_length, max_audio_duration, batch_size)
        print(f"Cache key: {cache_key}")
        
        # Try to load from cache
        print("\n=== Checking Cache ===")
        cached_dataset = load_dataset_cache(cache_key, cache_dir)
        
        if cached_dataset is not None:
            dataset = cached_dataset
            print("✅ Dataset loaded from cache!")
            dataset_time = 0
        else:
            # Build dataset
            print("\n=== Building Multimodal Dataset ===")
            dataset_start = time.time()
            
            dataset = build_multimodal_eagle3_dataset(
                data_path=data_path,
                processor=processor,
                chat_template=chat_template,
                max_length=max_length,
                max_audio_duration=max_audio_duration,
            )
            
            dataset_time = time.time() - dataset_start
            print(f"✅ Dataset built in {dataset_time:.2f}s")
            
            # Save to cache
            print("\n=== Saving Cache ===")
            save_dataset_cache(cache_key, dataset, cache_dir)
        
        # Test individual dataset items
        print(f"\n=== Dataset Information ===")
        print(f"Dataset size: {len(dataset)} items")
        
        # Test loading individual items
        test_items = min(3, len(dataset))
        successful_items = 0
        
        for i in range(test_items):
            if test_dataset_item(dataset, i):
                successful_items += 1
            else:
                break  # Stop on first failure
        
        print(f"Successfully loaded {successful_items}/{test_items} test items")
        
        if successful_items > 0:
            # Create DataLoader
            print("\n=== Creating DataLoader ===")
            dataloader_start = time.time()
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,  # Don't shuffle for testing
                collate_fn=multimodal_collate_fn,
                num_workers=0,  # Single process for testing
                pin_memory=False,
            )
            
            dataloader_time = time.time() - dataloader_start
            print(f"✅ DataLoader created in {dataloader_time:.3f}s")
            print(f"Number of batches: {len(dataloader)}")
            
            # Test batch loading
            if test_dataloader_batch(dataloader, batch_size):
                print("✅ DataLoader batch test passed")
            else:
                print("❌ DataLoader batch test failed")
        
        # Record final metrics
        end_time = time.time()
        end_memory = get_memory_usage()
        total_time = end_time - start_time
        
        print(f"\n=== Test Summary ===")
        print(f"✅ Test completed successfully!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Dataset preparation time: {dataset_time:.2f}s")
        print(f"End memory: {end_memory:.2f} MB")
        print(f"Memory increase: {end_memory - start_memory:.2f} MB")
        
        # Performance analysis
        if dataset_time > 0:
            items_per_second = len(dataset) / dataset_time
            print(f"Processing speed: {items_per_second:.2f} items/second")
        
        print(f"\n🎉 Multimodal data pipeline ready for Eagle3 training!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== Test Complete ===")


if __name__ == "__main__":
    # Suppress transformers warnings for cleaner output
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    main()