import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/roberta-base",
)
model = AutoModelForMaskedLM.from_pretrained(
    "FacebookAI/roberta-base",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)
inputs = tokenizer(
    "Plants create <mask> through a process known as photosynthesis.",
    return_tensors="pt",
).to("mps")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
