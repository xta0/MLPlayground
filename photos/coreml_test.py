import coremltools as ct
import numpy as np
from PIL import Image
from pathlib import Path

from torchvision import transforms

# === Paths ===
# MODEL_PATH = "FaceClassifierResnet18.mlpackage"
MODEL_PATH = "FaceClassifier.mlpackage"
LABELS_PATH = "classes.npy"
DEV_DIR = "/Volumes/dev-1t/photos/data/dev"

# === Load Core ML model ===
mlmodel = ct.models.MLModel(MODEL_PATH)
classes = np.load(LABELS_PATH)

# === Preprocessing (match Core ML input) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img)  # shape: (3, 224, 224), torch tensor
    np_input = tensor.numpy().astype(np.float32)
    np_input = np_input.reshape(1, 3, 224, 224)

    input_dict = {"input": np_input}
    output = mlmodel.predict(input_dict)
    output_probs = list(output.values())[0]  # Get the softmax output

    pred_idx = int(np.argmax(output_probs))
    return classes[pred_idx]

# === Evaluate ===
def evaluate_on_dev():
    total = 0
    correct = 0

    dev_dir = Path(DEV_DIR)
    for label_dir in dev_dir.iterdir():
        if not label_dir.is_dir():
            continue
        true_label = label_dir.name
        print("truth: ", true_label)
        for img_path in label_dir.glob("*.jpg"):
            if img_path.name.startswith("._"):
                continue
            try:
                pred_label = predict(img_path)
                print(pred_label)
                total += 1
                if pred_label == true_label:
                    correct += 1
                else:
                    print(f"❌ Wrong: {img_path.name} - predicted: {pred_label}, actual: {true_label}")
            except Exception as e:
                print(f"⚠️ Error on {img_path}: {e}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ Accuracy on dev set (Core ML): {accuracy:.2%} ({correct}/{total})")

if __name__ == "__main__":
    evaluate_on_dev()
