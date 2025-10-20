import os
import torch
import torchsummary
from torch import nn
from torchvision import datasets, transforms
from utils import show_images
import matplotlib.pyplot as plt

def prepare_cifar10_training_data(normalize=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)) if normalize else transforms.Lambda(lambda x: x)
    ])
    data_path = './CIFAR10_data/'
    trainset = datasets.CIFAR10(data_path, train=True,  download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    return trainloader

def prepare_cifar10_test_data(normalize=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)) if normalize else transforms.Lambda(lambda x: x)
    ])
    data_path = './CIFAR10_data/'
    testset = datasets.CIFAR10(data_path, train=False,  download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    return testloader

def prepare_mnist_training_data(normalize=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x)
    ])
    # Download and load the training data
    trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    return  trainloader


def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, with_act=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def transpose_conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0, with_act=True):
    modules = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding),
    ]
    if with_act:
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv_block(in_channels, 128) 
        self.conv2 = conv_block(128, 256)
        self.conv3 = conv_block(256, 512)
        self.conv4 = conv_block(512, 1024)
        self.linear = nn.Linear(4096, out_channels)

    def forward(self, x):
        bs = x.size(0)
        x = self.conv1(x) # (64, 3, 32, 32) -> (64, 128, 16, 16)
        x = self.conv2(x) # (64, 256, 8, 8)
        x = self.conv3(x) # (64, 512, 4, 4)
        x = self.conv4(x) # (64, 1024, 2, 2)
        x = x.view(bs, -1) # (64, 4096)
        x = self.linear(x) # (64, output_channels)
        return x
    

    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, 1024 * 2 * 2)
        self.deconv1 = transpose_conv_block(1024, 512)  # 2x2 -> 4x4
        self.deconv2 = transpose_conv_block(512, 256)   # 4x4 -> 8x8
        self.deconv3 = transpose_conv_block(256, 128)   # 8x8 -> 16x16
        self.deconv4 = transpose_conv_block(128, out_channels, with_act=False)  # 16x16 -> 32x32
        # self.out = nn.Sigmoid()

    def forward(self, x):
        bs = x.size(0)
        x = self.linear(x)
        x = x.view(bs, 1024, 2, 2)
        x = self.deconv1(x) # (64, 1024, 2, 2) -> (64, 512, 4, 4)
        x = self.deconv2(x) # (64, 512, 4, 4) -> (64, 256, 8, 8)
        x = self.deconv3(x) # (64, 256, 8, 8) -> (64, 128, 16, 16)
        x = self.deconv4(x) # (64, 128, 16, 16) -> (64, 3, 32, 32)
        # x = self.out(x)
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=16):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, out_channels=latent_dim)
        self.decoder = Decoder(in_channels=latent_dim, out_channels=in_channels)
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y



def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, device, epochs):
    num_epochs = epochs
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    trainLoader = prepare_cifar10_training_data()
    losses = []
    for epoch in range(num_epochs):
        loss = None
        for batch in trainLoader:
            optimizer.zero_grad()
            x = batch[0].to(device)
            y = model(x)
            loss = criterion(y, x)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    # draw the loss curve
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()


def test(model, device):
    testLoader = prepare_cifar10_test_data()
    model.to(device)  # Move model to the same device as input
    model.eval()
    
    with torch.no_grad():
        for batch in testLoader:
            x = batch[0].to(device)
            y = model(x)
            # compare x and y by showing images
            x = x.clamp(0,1).cpu().numpy().transpose(0, 2, 3, 1)
            y = y.clamp(0,1).cpu().numpy().transpose(0, 2, 3, 1)
            # unnormalize
            # mean = (0.4915, 0.4823, 0.4468)
            # std =  (0.2470, 0.2435, 0.2616)
            # x = (x * std) + mean
            # y = (y * std) + mean
            x = (x * 255).astype('uint8')
            y = (y * 255).astype('uint8')
            # show original and reconstructed images
            n = 8
            images = []
            titles = []
            for i in range(n):
                images.append(x[i])
                titles.append('Original')
                images.append(y[i])
                titles.append('Reconstructed')
            show_images(images, titles, cols=4, figsize=(12, 6))
            break


def main():
    trainLoader = prepare_cifar10_training_data()
    batch = next(iter(trainLoader))
    sample = batch[0]
    print(sample.shape)
    device = get_device()

    latent_dim = 512
    model = AutoEncoder(in_channels=3, latent_dim=latent_dim)
    # print the model architecture
    torchsummary.summary(model, input_size=(3, 32, 32))

    # Model Training
    model_file = f'autoencoder_cifar10_{latent_dim}.pth'
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
        print("Loaded pre-trained model.")
    else:
        epochs = 30
        model.to(device)
        train(model, device, epochs=epochs)
        model.to('cpu')
        # save the trained model
        torch.save(model.state_dict(), model_file)

    # Model Inference
    test(model, device = "cpu")

    
    


if __name__ == "__main__":
    main()