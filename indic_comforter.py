import torch
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_as_hybrid_rnnt_large"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.freeze()  # inference mode
model = model.to(device)  # transfer model to device

model.cur_decoder = "rnnt"
rnnt_text = model.transcribe(
    ["sample_audio_infer_ready.wav"], batch_size=1, language_id="as"
)[0]
print(rnnt_text)
