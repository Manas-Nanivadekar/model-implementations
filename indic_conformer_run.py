import os
import json
import torchaudio
import torch
from transformers import AutoModel  # trust_remote_code=True for Indic Conformer

# from datasets import load_metric


# Load the model on-the-fly from the Hub without pre-downloading a snapshot.
# This will download required files as needed into the default HF cache.
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
model.eval()


# -----------------------------
# Metrics
# -----------------------------
# wer_metric = load_metric("wer")


# -----------------------------
# Helpers
# -----------------------------
def get_reference_text(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Combine all transcript segments ignoring <PAUSE> or <UNINTELLIGIBLE>
    references = []
    for seg in data["transcriptions"]:
        text = seg["transcript"].strip()
        if text not in ["<PAUSE>", "<UNINTELLIGIBLE>"]:
            references.append(text)
    return " ".join(references)


def _load_audio_mono_16k(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # [C, T]
    # Convert to mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    else:
        wav = wav[:1]
    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav.squeeze(0), orig_freq=sr, new_freq=target_sr
        ).unsqueeze(0)
    return wav.contiguous()


# -----------------------------
# Function to run ASR on audio file (RNNT decoder by default)
# -----------------------------
def transcribe_audio(audio_file: str, lang: str = "hi", decoder: str = "rnnt") -> str:
    wav = _load_audio_mono_16k(audio_file).to(DEVICE)  # [1, T], 16k
    with torch.inference_mode():
        text = model(wav, lang, decoder)
    return text


# -----------------------------
# Files to process
# -----------------------------
json_files = ["Sample_Audio_1.json"]

all_references = []
all_predictions = []

for jf in json_files:
    print(f"\nProcessing {jf}...")
    # Get audio filename from JSON metadata
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)
    audio_filename = data["metadata"]["audio_filename"] + ".wav"
    lang = (
        data.get("metadata", {}).get("language") or os.environ.get("INDIC_LANG") or "hi"
    )

    reference_text = get_reference_text(jf)
    predicted_text = transcribe_audio(audio_filename, lang=lang, decoder="rnnt")

    print(f"Predicted: {predicted_text}")
    print(f"Reference: {reference_text}")

    all_predictions.append(predicted_text)
    all_references.append(reference_text)

# -----------------------------
# Compute WER
# -----------------------------
# overall_wer = wer_metric.compute(predictions=all_predictions, references=all_references)
# ?print(f"\nOverall WER: {overall_wer:.4f}")
