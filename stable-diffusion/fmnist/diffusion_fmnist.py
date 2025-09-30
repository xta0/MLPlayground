# ldm_fashion_mnist.py
import math, os, random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# -------------------------
# Device (MPS on Apple if available)
# -------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()
torch.manual_seed(42)
random.seed(42)

# -------------------------
# 1) Data
# -------------------------
# Fashion-MNIST is 28x28 grayscale; we resize to 32x32 for nicer powers-of-two downsampling.
def make_dataloaders(batch_size=64):
    tfm = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),              # [0,1]
        transforms.Normalize([0.5],[0.5])   # [-1,1]
    ])
    train = tv.datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
    test  = tv.datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False),
        DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False),
    )



# -------------------------
# 6) Main
# -------------------------
def main():
    os.makedirs("outputs", exist_ok=True)
    train_loader, test_loader = make_dataloaders(batch_size=128)
    sample = next(iter(train_loader))  # Warm-up for MPS
    print(sample[0].shape)


if __name__ == "__main__":
    main()
