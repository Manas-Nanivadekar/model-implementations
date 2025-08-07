import torch
from transformers import AutoModelForMaskedLM
from torch.export import export, Dim


from executorch.exir import to_edge_transform_and_lower

model = AutoModelForMaskedLM.from_pretrained(
    "FacebookAI/roberta-base", torch_dtype=torch.float32
).eval()


SEQ_LEN = 128
dummy_ids = torch.zeros((1, SEQ_LEN), dtype=torch.long)
dummy_attn = torch.ones_like(dummy_ids)
example_inputs = (dummy_ids, dummy_attn)
dynamic_shapes = {
    "input_ids": {1: Dim("seq", min=1, max=512)},
    "attention_mask": {1: Dim("seq", min=1, max=512)},
}

exported_program = export(model, example_inputs, dynamic_shapes=dynamic_shapes)

pte_program = to_edge_transform_and_lower(
    exported_program, partitioner=[]
).to_executorch()


with open("roberta-base-fillmask.pte", "wb") as f:
    f.write(pte_program.buffer)
