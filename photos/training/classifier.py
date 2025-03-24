import os
from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.preprocessing import LabelEncoder

# Load model
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def load_embeddings(data_dir):
    X = []
    y = []
    label_encoder = LabelEncoder()

    all_labels = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
    label_encoder.fit(all_labels)

    for label in all_labels:
        person_dir = Path(data_dir) / label
        for img_path in person_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            embedding = faces[0].embedding
            X.append(embedding)
            y.append(label)
    return np.array(X), label_encoder.transform(y), label_encoder

X, y, label_encoder = load_embeddings("data")
print(f"✅ Loaded {len(X)} face embeddings across {len(label_encoder.classes_)} identities")
