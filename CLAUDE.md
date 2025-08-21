# CLAUDE.md

### Core Rules

1. **Don't write code until I tell you to.** I need to check the logic first.
2. **Only write the code I ask for.** Don't add extra features.
3. **If you get an error, stop and ask me.** This prevents getting stuck.
4. **Tell me if you are unsure about anything.** This is to avoid risks.

### Goal

The EAGLE-3 project currently only supports text. We want to add support for large audio models.

- **Audio Model Reference:** If you need, you can read `specforge/modeling/target/qwen2_audio.py`
- **Training Method Reference:** SpecForge uses online-training to build a draft model for a target model. I prepare a summary `online-training-summary.md`.

### To-Do

#### Implementation
- [x] Implement `Qwen2AudioForConditionalGeneration` as target model
- [x] Create `AudioOnlineEagle3Model` extending `OnlineEagle3Model`
- [x] Implement audio input handling in forward pass
- [x] Add audio feature extraction and processing methods
- [x] Modify Qwen2Audio to return `inputs_embeds` for draft model
- [x] Configure `LlamaForCausalLMEagle3` as draft model with Qwen2-Audio dimensions
- [ ] Modify `build_multimodal_eagle3_dataset` for audio inputs
- [ ] Add audio preprocessing pipeline (`load_audio_file`, feature extraction)
- [ ] Create Qwen2-Audio chat template for multimodal conversations
- [ ] Update training script to use `AudioOnlineEagle3Model`
- [ ] Add audio data loading and batching to training loop
- [ ] Implement multimodal loss computation (text + audio tokens)
- [ ] Add audio-specific validation during training

#### Testing
- [x] Create basic test for audio target model loading and inference
- [x] Create test for draft model and AudioOnlineEagle3Model integration
- [x] Validate AudioOnlineEagle3Model forward pass with audio inputs
- [x] Test multimodal loss computation and accuracy calculation
- [ ] Write integration test for full multimodal training pipeline
- [ ] Add test cases for audio preprocessing functions
- [ ] Create end-to-end test with sample audio input
- [ ] Add performance benchmarks (audio vs text-only training)
- [ ] Validate memory usage and GPU utilization with audio models