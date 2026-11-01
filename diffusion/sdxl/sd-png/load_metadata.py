from PIL import Image

OUTPUT_DIR = "./output_dir"

image = Image.open(f"{OUTPUT_DIR}/output_with_metadata.png")

metadata = image.info

# print the meta
for key, value in metadata.items():
    print(f"{key}: {value}")