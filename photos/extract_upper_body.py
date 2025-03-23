import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import os

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection

def extract_upper_body(image_path, output_dir, final_size=256, top_padding=160, side_padding=40):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        return 0

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face detection (for top boundary)
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_det:
        face_result = face_det.process(img_rgb)

    if not face_result.detections:
        print(f"⚠️ No face detected in {image_path.name}")
        return 0

    # Get face top
    face_bbox = face_result.detections[0].location_data.relative_bounding_box
    face_top_y = int(face_bbox.ymin * h)
    crop_top = max(face_top_y - top_padding, 0)

    # Pose detection (for bottom boundary)
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        pose_result = pose.process(img_rgb)

    if not pose_result.pose_landmarks:
        print(f"⚠️ No pose detected in {image_path.name}")
        return 0

    landmarks = pose_result.pose_landmarks.landmark
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Use max y of both hips for lower body boundary
    hip_y = int(max(left_hip.y, right_hip.y) * h)
    crop_bottom = min(hip_y + side_padding, h)

    # Crop height
    crop_height = crop_bottom - crop_top
    crop_width = crop_height  # square crop

    # Horizontal center: use face center
    face_center_x = int((face_bbox.xmin + face_bbox.width / 2) * w)
    crop_left = max(face_center_x - crop_width // 2, 0)
    crop_right = min(crop_left + crop_width, w)

    # Clamp left again if right edge overflows
    if crop_right - crop_left < crop_width:
        crop_left = max(crop_right - crop_width, 0)

    # Final crop and resize
    cropped = img[crop_top:crop_bottom, crop_left:crop_right]
    resized = cv2.resize(cropped, (final_size, final_size), interpolation=cv2.INTER_AREA)

    out_path = output_dir / f"{image_path.stem}_upper_body.jpg"
    cv2.imwrite(str(out_path), resized)
    print(f"✅ Saved combined upper body from {image_path.name} -> {out_path.name}")
    return 1

def process_directory(source_dir, output_dir):
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0
    for root, _, files in os.walk(source_path):
        for name in files:
            file_path = Path(root) / name
            if (
                file_path.suffix.lower() == ".jpg"
                and not file_path.name.startswith(".")
                and not file_path.name.startswith("._")
            ):
                try:
                    total += extract_upper_body(file_path, output_path)
                except Exception as e:
                    print(f"❌ Failed on {file_path}: {e}")
            else:
                print(f"🧾 Skipping hidden or unsupported file: {file_path.name}")

    print(f"\n🎉 Done! Extracted {total} upper-body crops with face+pose into {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_combined_upper_body.py <source_dir> <output_dir>")
        sys.exit(1)

    process_directory(sys.argv[1], sys.argv[2])
