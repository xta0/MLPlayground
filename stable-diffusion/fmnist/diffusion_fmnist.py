import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from torch import optim
import deepinv

device = "mps"

batch_size = 32
image_size = 32 

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))
])
# Download and load the training data
trainset = datasets.FashionMNIST('./F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('./F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

lr = 1e-4
epochs = 10

model = deepinv.models.DiffUNet(
    in_channels=1,
    out_channels=1,
    pretrained=None
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

loss_fn = deepinv.loss.MSE()