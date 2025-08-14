from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from huggingface_hub import login

login("YOUR_HF_Token")

# Load model and processor
device = "cuda"
processor = WhisperProcessor.from_pretrained(
    "jiviai/audioX-south-v1",
)
model = WhisperForConditionalGeneration.from_pretrained("jiviai/audioX-south-v1").to(
    device
)
model.config.forced_decoder_ids = None

# Load and preprocess audio
audio_path = "audio.wav"
audio_np, sr = librosa.load(audio_path, sr=None)
if sr != 16000:
    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

input_features = (
    processor(audio_np, sampling_rate=16000, return_tensors="pt")
    .to(device)
    .input_features
)

# Generate predictions
# Use ISO 639-1 language codes: "hi", "mr", "gu" for North; "ta", "te", "kn", "ml" for South
# Or omit the language argument for automatic language detection
predicted_ids = model.generate(input_features, task="transcribe", language="hi")

# Decode predictions
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
