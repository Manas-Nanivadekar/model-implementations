from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch

device = "cpu"
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(
    repo_id, torch_dtype=torch.bfloat16, device_map=device
)

inputs = processor.apply_transcription_request(
    language="hi",
    audio="audio.mp4",
    model_id=repo_id,
)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(
    outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
)

print("\nGenerated responses:")
print("=" * 80)
for decoded_output in decoded_outputs:
    print(decoded_output)
    print("=" * 80)
