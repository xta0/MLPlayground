import os
from PIL import Image
import json

# Directories
input_dir = "./tao_hi_res"          # Folder with high-res images
output_dir = "./tao"      # Where cropped and renamed images will go
os.makedirs(output_dir, exist_ok=True)

# Output metadata file
metadata_path = os.path.join(output_dir, "metadata.jsonl")

# Desired image size (square)
output_size = 512  # Use 1024 for SDXL, 512 for SD 1.5

# Center crop and resize helper
def center_crop_square(image):
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return image.crop((left, top, right, bottom)).resize((output_size, output_size), Image.LANCZOS)

# Get all .jpg files sorted
jpg_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])

with open(metadata_path, "w") as meta_file:
    for idx, jpg_file in enumerate(jpg_files, start=1):
        new_name = f"{idx:03d}.png"
        img_path = os.path.join(input_dir, jpg_file)
        new_img_path = os.path.join(output_dir, new_name)

        # Load, crop, resize, and save
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_cropped = center_crop_square(img)
            img_cropped.save(new_img_path, "PNG")

        # Write metadata
        json_line = {"file_name": new_name, "text": "a man named Tao"}
        meta_file.write(json.dumps(json_line) + "\n")

print(f"✅ Center-cropped and processed {len(jpg_files)} images. Metadata saved to {metadata_path}")
