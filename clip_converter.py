import torch
import coremltools as ct
from transformers import AutoProcessor, AutoModel


MODEL_ID = "openai/clip-vit-base-patch32"
model = AutoModel.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()


dummy_pixel_values = torch.rand(1, 3, 224, 224)

dummy_input_ids = torch.randint(0, 1000, (3, 77))


class CLIPClassifer(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, pixel_values, input_ids):

        outputs = self.model(pixel_values=pixel_values, input_ids=input_ids)

        return outputs.logits_per_image


wrapped_model = CLIPClassifer(model)


traced_model = torch.jit.trace(wrapped_model, (dummy_pixel_values, dummy_input_ids))

mlmodel = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(name="pixel_values", shape=dummy_pixel_values.shape, dtype=float),
        ct.TensorType(name="input_ids", shape=dummy_input_ids.shape, dtype=int),
    ],
    outputs=[ct.TensorType(name="logits_per_image")],
    compute_units=ct.ComputeUnit.ALL,  # Use CPU, GPU, and Neural Engine
)


output_path = "CLIPClassifier.mlpackage"
mlmodel.save(output_path)
