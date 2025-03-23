import os
import random
import shutil
from pathlib import Path
import sys

def move_random_subset(input_dir, output_dir, ratio=0.1, image_exts={'.jpg', '.jpeg', '.png'}):
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.exists():
        print(f"❌ Training folder does not exist: {input_path}")
        return

    all_images = [
        p for p in input_path.rglob("*")
        if p.suffix.lower() in image_exts
        and p.is_file()
        and not p.name.startswith("._")
    ]

    random.shuffle(all_images)

    num_to_move = max(1, int(len(all_images) * ratio))
    images_to_move = all_images[:num_to_move]

    print(f"📦 Found {len(all_images)} images. Moving {num_to_move} to {output_path}...")

    for img in images_to_move:
        relative_path = img.relative_to(input_path)
        target_path = output_path / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img), str(target_path))
        print(f"✅ Moved: {relative_path}")

    print(f"\n🎉 Done! Moved {num_to_move} images to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python move_random_subset.py <input_dir> <output_dir>")
        sys.exit(1)

    move_random_subset(sys.argv[1], sys.argv[2])
