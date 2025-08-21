"""
Test script for Qwen2-Audio with LlamaForCausalLMEagle3 draft model and AudioOnlineEagle3Model integration.
"""

import unittest
import torch
import json
import librosa
from transformers import AutoProcessor, LlamaConfig

from specforge.modeling.target.qwen2_audio import Qwen2AudioForConditionalGeneration
from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
from specforge.core.eagle3_multimodal import AudioOnlineEagle3Model


class TestQwen2AudioDraftModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "/cpfs04/user/lvqidan/speechllm/checkpoints/Qwen2-Audio-7B-Instruct"
        self.demo_path = "/cpfs04/user/lvqidan/SpecForge/cache/dataset/gigaspeech_demo.jsonl"
        self.config_path = "/cpfs04/user/lvqidan/SpecForge/configs/qwen2-audio-eagle3.json"
        
        # Load demo data
        with open(self.demo_path) as f:
            self.data = [json.loads(line) for line in f.readlines()]
        
        # Load draft config from JSON file
        with open(self.config_path) as f:
            self.draft_config_dict = json.load(f)
     
    def test_audio_online_eagle3_integration(self):
        """Test AudioOnlineEagle3Model with target and draft models."""
        
        # Initialize target model
        print("Loading target model...")
        target_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        # Initialize draft model with LlamaForCausalLMEagle3
        print("Creating draft model...")
        draft_config = LlamaConfig(**self.draft_config_dict)
        draft_model = LlamaForCausalLMEagle3(draft_config).to("cuda").to(torch.bfloat16)
        draft_model.t2d[:draft_config.draft_vocab_size] = True
        # Create AudioOnlineEagle3Model
        print("Creating AudioOnlineEagle3Model...")
        eagle_model = AudioOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            length=3,  # Shorter TTT for testing
        )
        
        print("✅ AudioOnlineEagle3Model created successfully")
        
        # Prepare test input
        processor = AutoProcessor.from_pretrained(self.model_path)
        
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": self.data[0]['wav_path']},
                {"type": "text", "text": "Recognize the speech and give me the transcription."},
            ]},
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio, _ = librosa.load(
            self.data[0]['wav_path'],
            sr=processor.feature_extractor.sampling_rate
        )
        
        inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True)
        
        # Test forward pass
        print("Testing forward pass...")
        loss_mask = torch.ones_like(inputs.input_ids, dtype=torch.long).to("cuda")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        input_features = inputs.input_features.to("cuda")
        feature_attention_mask = inputs.feature_attention_mask.to("cuda")
        
        with torch.no_grad():
            plosses, vlosses, acces = eagle_model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                loss_mask = loss_mask,
                input_features = input_features,
                feature_attention_mask = feature_attention_mask
            )
        
        # Check outputs
        self.assertEqual(len(plosses), 3)  # Should have 3 losses (TTT length)
        self.assertEqual(len(acces), 3)  # Should have 3 accuracy values
        
        print(f"✅ Forward pass successful!")
        print(f"   Loss values: {[loss.item() for loss in plosses]}")
        print(f"   Accuracy values: {acces}")
    


if __name__ == "__main__":
    # Run specific test or all tests
    suite = unittest.TestSuite()
    
    # Add tests in order
    suite.addTest(TestQwen2AudioDraftModel('test_audio_online_eagle3_integration'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)