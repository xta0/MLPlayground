import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from pathlib import Path

# === Config ===
MODEL_PATH = "face_classifier_resnet18.pt"
LABELS_PATH = "classes.npy"
DEV_DIR = "/Volumes/dev-1t/photos/data/dev"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === Image preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    classes = np.load(LABELS_PATH)
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, len(classes)),
        nn.Softmax(dim=1)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model, classes

def predict(model, img_path, classes):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        return classes[pred_idx]

def evaluate_on_dev():
    model, classes = load_model()
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
                pred_label = predict(model, img_path, classes)
                print(pred_label)
                total += 1
                if pred_label == true_label:
                    correct += 1
                else:
                    print(f"❌ Wrong: {img_path.name} - predicted: {pred_label}, actual: {true_label}")
            except Exception as e:
                print(f"⚠️ Error on {img_path}: {e}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ Accuracy on dev set: {accuracy:.2%} ({correct}/{total})")

if __name__ == "__main__":
    evaluate_on_dev()