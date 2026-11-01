import torch
from transformers import AutoTokenizer, T5EncoderModel, CLIPTextModelWithProjection

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print("device:", device)

prompt = "a photo of a corgi wearing sunglasses"

# ---- T5 (token sequence) ----
t5_name = "t5-base"
t5_tok = AutoTokenizer.from_pretrained(t5_name)
t5_enc = T5EncoderModel.from_pretrained(t5_name).to(device).eval()

t5_inputs = t5_tok(
    prompt,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=64,
).to(device)

with torch.no_grad():
    t5_out = t5_enc(**t5_inputs)
t5_hidden = t5_out.last_hidden_state  # [B, Nt, dt5]
print("\nT5 last_hidden_state:", t5_hidden.shape, t5_hidden.dtype)

# ---- CLIP (pooled vector) via safetensors repo ----


clip_name = "openai/clip-vit-base-patch32"

clip_tok = AutoTokenizer.from_pretrained(clip_name)
clip_txt = CLIPTextModelWithProjection.from_pretrained(
    clip_name,
    use_safetensors=True,          # <-- key line
    torch_dtype=torch.float32,      # keep simple for MPS/CPU
).to(device).eval()

clip_inputs = clip_tok(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    clip_out = clip_txt(**clip_inputs)

clip_last = clip_out.last_hidden_state     # [B, Nclip, d_model]
clip_pooled = clip_out.text_embeds         # [B, projection_dim]
print("CLIP last_hidden_state:", clip_last.shape)
print("CLIP pooled text_embeds:", clip_pooled.shape)



# ---- Build ids ----
latent_h, latent_w = 64, 64
patch = 2
hi, wi = latent_h // patch, latent_w // patch
Ni = hi * wi
Nt = t5_hidden.shape[1]

text_ids = torch.zeros((Nt, 3), device=device, dtype=torch.float32)

ys = torch.arange(hi, device=device, dtype=torch.float32)
xs = torch.arange(wi, device=device, dtype=torch.float32)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
image_ids = torch.stack([torch.ones_like(yy), yy, xx], dim=-1).view(Ni, 3)

ids = torch.cat([text_ids, image_ids], dim=0)

print("\ntext_ids:", text_ids.shape, "unique rows:", torch.unique(text_ids, dim=0).shape[0])
print("image_ids:", image_ids.shape, "first 5:\n", image_ids[:5])
print("ids concat:", ids.shape, "last 5:\n", ids[-5:])
