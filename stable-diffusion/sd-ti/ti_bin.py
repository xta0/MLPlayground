import torch
loaded_leared_embeds = torch.load('/Volumes/ai-1t/ti/midjourney_style.bin', map_location='cpu')
keys = list(loaded_leared_embeds.keys())
for key in keys:
    print(key, ": ", loaded_leared_embeds[key].shape) # <midjourney-style> :  torch.Size([768])
