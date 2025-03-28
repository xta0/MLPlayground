import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

# === Config ===
TRAINING_DIR = "/Volumes/dev-1t/photos/data/training"
MODEL_PATH = "face_classifier_resnet18.pt"
LABELS_PATH = "classes.npy"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
PATIENCE = 5  # early stopping patience
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === Image preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_dataset(data_dir):
    X, y = [], []
    label_encoder = LabelEncoder()

    data_dir = Path(data_dir)
    all_labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    label_encoder.fit(all_labels)

    for label in all_labels:
        person_dir = data_dir / label
        for img_path in person_dir.glob("*.jpg"):
            if img_path.name.startswith("._"):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                X.append(img_tensor)
                y.append(label)
            except:
                print(f"⚠️ Failed to load image: {img_path}")
    return torch.stack(X), label_encoder.transform(y), label_encoder

class FaceDataset(Dataset):
    def __init__(self, images, labels):
        self.X = images
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    # === Load Data ===
    X, y, label_encoder = load_dataset(TRAINING_DIR)
    print(f"Loaded {len(X)} images across {len(label_encoder.classes_)} classes")

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    train_loader = DataLoader(FaceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FaceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # === Load Pretrained Model ===
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(label_encoder.classes_)),
        nn.Softmax(dim=1)
    )
    model = model.to(DEVICE)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            # Forward pass => probabilities
            probs = model(batch_x)  # shape [batch_size, num_classes], sums to 1
            # Convert probabilities to log-probabilities for NLLLoss
            log_probs = torch.log(probs + 1e-8)
            loss = criterion(log_probs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(DEVICE), val_y.to(DEVICE)
                preds = model(val_x)
                predicted = torch.argmax(preds, dim=1)
                correct += (predicted == val_y).sum().item()
                total += val_y.size(0)
        val_accuracy = correct / total

        print(f"📦 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Val Acc: {val_accuracy:.2%}")

        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': len(label_encoder.classes_)
            }, MODEL_PATH)
            np.save(LABELS_PATH, label_encoder.classes_)
            print("💾 Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == "__main__":
    main()
