from datasets import load_dataset
from utils import show_images
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn


def prepare_training_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Download and load the training data
    trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    return  trainloader


def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.conv1 = conv_block(in_channels, 128) 
        self.conv2 = conv_block(128, 256)  # (64, 256, 7, 7)
        self.conv3 = conv_block(256, 512)  # (64, 512, 3, 3)
        self.conv4 = conv_block(512, 1024) # (64, 1024, 1, 1)
        self.linear = nn.Linear(1024, out_channels)

    def forward(self, x):
        x = self.conv1(x) # (64, 1, 28, 28) -> (64, 128, 14, 14)
        x = self.conv2(x) # (64, 256, 7, 7)
        x = self.conv3(x) # (64, 512, 3, 3)
        x = self.conv4(x) # (64, 1024, 1, 1)
        x = x.view(64, -1) # (64, 1024)
        x = self.linear(x) # (64, 16)
        return x

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    trainLoader = prepare_training_data()
    batch = next(iter(trainLoader))
    sample = batch[0]
    print(sample.shape)
    encoder = Encoder()
    encoder.eval()
    y = encoder(sample)
    print(y.shape)

    


if __name__ == "__main__":
    main()