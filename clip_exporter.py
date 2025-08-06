from transformers import AutoTokenizer
import os

# Define the model name
model_name = "openai/clip-vit-base-patch32"

# Define the output directory
output_dir = "./clip_exporter"
os.makedirs(output_dir, exist_ok=True)

# Load the tokenizer
print("Downloading tokenizer files from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the necessary files
tokenizer.save_pretrained(output_dir)

print(f"\nSuccess! Tokenizer files saved to a new folder named '{output_dir}'")
print("You now need to drag 'vocab.json' and 'merges.txt' from that folder into Xcode.")
