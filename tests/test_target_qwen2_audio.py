import unittest
from specforge.modeling.target.qwen2_audio import Qwen2AudioForConditionalGeneration
from transformers import AutoProcessor
import json
import librosa

class TestAutoModelForCausalLM(unittest.TestCase):

    def test_automodel(self):
        """init"""
        model_path = "/cpfs04/user/lvqidan/speechllm/checkpoints/Qwen2-Audio-7B-Instruct"
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="auto")
        demo_path = "/cpfs04/user/lvqidan/SpecForge/cache/dataset/gigaspeech_demo.jsonl"
        
        with open(demo_path) as f:
            data = [json.loads(line) for line in f.readlines()]
        
        print(data[0])

        conversation = [
        {"role": "user", "content": [
        {"type": "audio", "audio_url": data[0]['wav_path']},
        {"type": "text", "text": "Recognize the speech and give me the transcription."},
        ]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                            ele['audio_url'], 
                            sr=processor.feature_extractor.sampling_rate)[0]
                        )
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs.input_ids = inputs.input_ids.to("cuda")
        generate_ids = model.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(response)


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(TestAutoModelForCausalLM))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
