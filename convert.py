import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import coremltools as ct

# --- 1. Load the whisper-tiny model from Hugging Face ---
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# --- 2. Create sample inputs required for the conversion ---
# The model's encoder expects mel-spectrograms of the audio
mel_input = torch.rand(1, model.config.num_mel_bins, 3000)

# The model's decoder expects token IDs
decoder_input_ids = torch.tensor([[50258, 50259, 50359]])  # Example start tokens

# --- 3. Convert the Audio Encoder ---
# This part of the model processes the audio spectrogram
traced_encoder = torch.jit.trace(model.get_encoder(), mel_input)

encoder_mlmodel = ct.convert(
    traced_encoder,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=mel_input.shape, name="mel_input")],
    compute_units=ct.ComputeUnit.ALL,
)
encoder_mlmodel.save("WhisperEncoder.mlpackage")
print("✅ Encoder saved to WhisperEncoder.mlpackage")


# --- 4. Convert the Text Decoder ---
# This part of the model generates the text transcription
# It needs both the encoder's output and the current text tokens
encoder_output = traced_encoder(mel_input)

traced_decoder = torch.jit.trace(
    model, (encoder_output, decoder_input_ids), strict=False
)

decoder_mlmodel = ct.convert(
    traced_decoder,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(shape=encoder_output.shape, name="encoder_output"),
        ct.TensorType(
            shape=decoder_input_ids.shape, dtype=int, name="decoder_input_ids"
        ),
    ],
    compute_units=ct.ComputeUnit.ALL,
)
decoder_mlmodel.save("WhisperDecoder.mlpackage")
print("✅ Decoder saved to WhisperDecoder.mlpackage")
