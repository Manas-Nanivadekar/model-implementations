from transformers import AutoModel
import torch
import torchaudio
import soundfile as sf
import numpy as np


def load_audio_mono_16k(path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    wav = torch.from_numpy(audio)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)

    wav = wav.unsqueeze(0).contiguous()
    return wav, target_sr


device = "cpu"


model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
).to(device)
model.eval()


wav, sr = load_audio_mono_16k("audio.flac", target_sr=16000)  # -> [1, T], 16k
wav = wav.to(device)


lang = "hi"

with torch.inference_mode():
    transcription_ctc = model(wav, lang, "ctc")
    with open("ctc.txt", "w", encoding="utf-8") as f:
        f.write(transcription_ctc)
    print("CTC Transcription:", transcription_ctc)

    transcription_rnnt = model(wav, lang, "rnnt")
    with open("rnnt.txt", "w", encoding="utf-8") as f:
        f.write(transcription_rnnt)
    print("RNNT Transcription:", transcription_rnnt)
