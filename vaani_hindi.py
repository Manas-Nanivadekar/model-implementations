import torch
from transformers import pipeline

audio = "audio.mp3"
device = "cpu"
modelTags = "ARTPARK-IISc/whisper-small-vaani-hindi"
transcribe = pipeline(
    task="automatic-speech-recognition",
    model=modelTags,
    chunk_length_s=30,
    device=device,
)
transcribe.model.config.forced_decoder_ids = (
    transcribe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")
)

print("Transcription: ", transcribe(audio)["text"])
