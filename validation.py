import onnxruntime as ort
import soundfile as sf
import numpy as np

pcm, sr = sf.read("combined.wav", dtype="float32")
assert sr == 16000
if pcm.ndim == 2:
    pcm = pcm.mean(axis=1)
sess = ort.InferenceSession("preproc_fbank.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {"pcm": pcm[None, :]})[0]  # [1, n_frames, 80] (static if Fix C)
print(out.shape, out.dtype)
