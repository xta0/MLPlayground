import torch
import open_clip

# Load OpenCLIP G/14 Model
model, preprocess, _ = open_clip.create_model_and_transforms(
    "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
)
tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

text = ["a running dog"]  # OpenCLIP expects a list of strings

# Tokenize text
input_tokens = tokenizer(text)
print("OpenCLIP Token IDs:", input_tokens.shape)

# Encode token ids to embeddings
with torch.no_grad():
    text_embeds = model.encode_text(input_tokens)

print("OpenCLIP Embedding Shape:", text_embeds.shape)  # (1, 1408) for G/14 model
print(text_embeds)  # Display the embedding values