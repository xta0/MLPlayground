import cv2
import mediapipe as mp
from pathlib import Path
import numpy as np
import os

def extract_faces_mediapipe(image_path, output_dir, final_size=256, top_padding=120, bottom_padding=40):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        return 0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(img_rgb)

    if not results.detections:
        print(f"⚠️ No faces detected in {image_path.name}")
        return 0

    count = 0
    for i, detection in enumerate(results.detections):
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        face_top = y
        face_bottom = y + bh
        face_left = x
        face_right = x + bw

        face_height = bh

        # Calculate crop region
        crop_height = face_height + top_padding + bottom_padding
        crop_width = crop_height  # enforce square

        face_center_x = (face_left + face_right) // 2
        crop_left = face_center_x - crop_width // 2
        crop_right = crop_left + crop_width

        crop_top = face_top - top_padding
        crop_bottom = crop_top + crop_height

        # Clamp to image bounds
        crop_left = max(0, crop_left)
        crop_right = min(w, crop_right)
        crop_top = max(0, crop_top)
        crop_bottom = min(h, crop_bottom)

        cropped = img[crop_top:crop_bottom, crop_left:crop_right]
        print(cropped.shape)
        resized = cv2.resize(cropped, (final_size, final_size), interpolation=cv2.INTER_AREA)

        face_filename = output_dir / f"{image_path.stem}_face_{i+1}.jpg"
        cv2.imwrite(str(face_filename), resized)
        print(f"✅ Saved face {i+1} from {image_path.name} -> {face_filename.name}")
        count += 1

    return count

def process_directory(source_dir, output_dir):
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    total_faces = 0
    for root, _, files in os.walk(source_path):
        for name in files:
            file_path = Path(root) / name
            if (
                file_path.suffix.lower() == ".jpg"
                and not file_path.name.startswith(".")
                and not file_path.name.startswith("._")
            ):
                try:
                    total_faces += extract_faces_mediapipe(file_path, output_path)
                except Exception as e:
                    print(f"❌ Failed on {file_path}: {e}")
            else:
                print(f"🧾 Skipping hidden or unsupported file: {file_path.name}")

    print(f"\n🎉 Done! Extracted {total_faces} face(s) into {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_faces.py <source_dir> <output_dir>")
        sys.exit(1)

    process_directory(sys.argv[1], sys.argv[2])
