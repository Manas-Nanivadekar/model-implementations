import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForMaskedLM, AutoTokenizer


# Define a wrapper class to handle the model's dictionary output
class RobertaMaskedLMWrapper(nn.Module):
    def __init__(self, model):
        super(RobertaMaskedLMWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # Pass the inputs to the underlying RoBERTa model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Extract and return only the logits
        return outputs.logits


def convert_roberta_to_coreml():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    base_model = AutoModelForMaskedLM.from_pretrained(
        "FacebookAI/roberta-base",
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    base_model.eval()

    model = RobertaMaskedLMWrapper(base_model)
    model.eval()

    print("Preparing example inputs for tracing...")
    text = "Plants create <mask> through a process known as photosynthesis."

    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=128)
    example_input_ids = inputs["input_ids"]
    example_attention_mask = inputs["attention_mask"]

    print("Tracing the model...")
    traced_model = torch.jit.trace(
        model, (example_input_ids, example_attention_mask), strict=False
    )

    input_ids_spec = ct.TensorType(
        name="input_ids",
        shape=(1, ct.RangeDim(1, 512)),
        dtype=int,
    )
    attention_mask_spec = ct.TensorType(
        name="attention_mask",
        shape=(1, ct.RangeDim(1, 512)),
        dtype=int,
    )

    print("Converting the model to Core ML format...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[input_ids_spec, attention_mask_spec],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.iOS15,
    )

    output_path = "RoBERTa.mlpackage"
    print(f"Saving the Core ML model to {output_path}...")
    coreml_model.save(output_path)
    print("Conversion complete!")


if __name__ == "__main__":
    convert_roberta_to_coreml()
