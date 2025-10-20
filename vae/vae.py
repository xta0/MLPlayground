import os
import json
from networkx import sigma
import torch
import torchsummary
import torchvision
from torch import nn
from torchvision import datasets, transforms
from utils import show_images
import matplotlib.pyplot as plt

def prepare_cifar10_training_data(normalize=False):
    transform = transforms.Compose([
        transforms.ToTensor(), # converts the pixel to [0, 1]
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


def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalization=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
    ]
    if normalization:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def transpose_conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0, with_act=True):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding),
    ]
    if with_act:
        layers += [nn.BatchNorm2d(out_channels), nn.ReLU()]
    return nn.Sequential(*layers)

def upsample_block(in_channels, out_channels):
    # Upsample + Conv avoids deconv blur/checkerboard
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        nn.LeakyReLU(0.2, inplace=True),
    )

def linear_block(in_features, out_features, relu=False):
    layers = [
        nn.Linear(in_features, out_features),
    ]
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class VAEEncoder(nn.Module):
    """
    32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
    Channels: 3 -> 64 -> 128 -> 256 -> feature_channels (e.g., 512)
    """
    def __init__(self, in_channels, feature_channels):
        super().__init__()
        self.conv1 = conv_block(in_channels, feature_channels // 8) # (B, 3, 32, 32) -> (B, 64, 16, 16)
        self.conv2 = conv_block(feature_channels // 8, feature_channels // 4) # (B, 64, 16, 16) -> (B, 128, 8, 8)
        self.conv3 = conv_block(feature_channels // 4, feature_channels // 2) # (B, 128, 8, 8) -> (B, 256, 4, 4)
        self.conv4 = conv_block(feature_channels // 2, feature_channels) # (B, 256, 4, 4) -> (B, feature_channels, 2, 2)

    def forward(self, x):
        x = self.conv1(x) # (64, 3, 32, 32) -> (64, 64, 16, 16)
        x = self.conv2(x) # (64, 64, 16, 16) -> (64, 128, 8, 8)
        x = self.conv3(x) # (64, 128, 8, 8) -> (64, 256, 4, 4)
        x = self.conv4(x) # (64, 256, 4, 4) -> (64, 256, 2, 2)
        return x
    
class VAEDecoder(nn.Module):
    """
    2x2 -> 4x4 -> 8x8 -> 16x16 -> 32x32
    Channel path mirrors encoder.
    """
    def __init__(self, feature_channels, out_channels):
        super().__init__()
        self.up1 = upsample_block(feature_channels, feature_channels // 2)  # 2x2 -> 4x4
        self.up2 = upsample_block(feature_channels // 2, feature_channels // 4)   # 4x4 -> 8x8
        self.up3 = upsample_block(feature_channels // 4, feature_channels // 8)   # 8x8 -> 16x16
        self.up4 = upsample_block(feature_channels // 8, feature_channels // 16)   # 16x16 -> 32x32
        self.conv_out = nn.Conv2d(feature_channels // 16, out_channels, kernel_size=3, padding=1, bias=True)
        self.out = nn.Sigmoid() # ensure output is in [0, 1]

    def forward(self, x):
        x = self.up1(x) # (B, 512, 2, 2) -> (B, 256, 4, 4)
        x = self.up2(x) # (B, 256, 4, 4) -> (B, 128, 8, 8)
        x = self.up3(x) # (B, 128, 8, 8) -> (B, 64, 16, 16)
        x = self.up4(x) # (B, 64, 16, 16) -> (B, 32, 32, 32)
        x = self.conv_out(x) # (B, 32, 32, 32) -> (B, 3, 32, 32)
        x = self.out(x)
        return x

class VAE(nn.Module):
    def __init__(self, in_channels=3, 
                 latent_dim=256, 
                 feature_size=2, 
                 feature_channels=512):
        super().__init__()
        self.encoder = VAEEncoder(in_channels=in_channels, feature_channels=feature_channels)
        self.decoder = VAEDecoder(feature_channels=feature_channels, out_channels=in_channels)

        self.latent_dim = latent_dim
        self.feature_size = feature_size
        self.feature_channels = feature_channels
        self.feature_volume = self.feature_channels * self.feature_size * self.feature_size

        self.mu         = linear_block(self.feature_volume, self.latent_dim)
        self.logvar     = linear_block(self.feature_volume, self.latent_dim)
        self.projection = linear_block(self.latent_dim, self.feature_volume)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def q(self, x):
        # x comes in as (B, Cfeat, 2, 2) from encoder
        x = x.view(x.size(0), -1)  # flatten
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.q(x)
        z = self.z(mu, logvar)
        z_projected = self.projection(z).view(
            x.size(0),
            self.feature_channels,
            self.feature_size,
            self.feature_size
        )
        y = self.decoder(z_projected)

        return y, mu, logvar

def vae_loss_mse(x, recon_x, mu, logvar):
    # Reconstruction loss
    bs = x.size(0)
    x = x.view(bs, -1)
    recon_x = recon_x.view(bs, -1)
    # per image loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='none').sum(dim=-1) #[Batch]
    
    # KL: sum over latent dim
    # mu, logvar shape: [B, D]
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    loss = (recon_loss + kld_per_sample).mean(dim=0)
    return loss, recon_loss, kld_per_sample

def vae_loss_mse2(x, recon_x, mu, logvar, beta=1e-3):
    # per-image mean MSE (size-invariant)
    per_img_mse = torch.nn.functional.mse_loss(
        recon_x, x, reduction='none'
    ).mean(dim=(1,2,3))                           # [B]

    kld_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1
    )                                             # [B]

    per_sample = per_img_mse + beta * kld_per_sample
    return per_sample.mean(), per_img_mse, kld_per_sample

def vae_loss_bce(x, recon_x, mu, logvar):
    # Reconstruction loss
    recon_loss = nn.BCELoss(size_average=False)(recon_x, x) / x.size(0)
    # KL divergence
    kld_loss = ((mu**2 + logvar.exp() - 1 - logvar) / 2).mean()
    
    loss = recon_loss + kld_loss
    return loss, recon_loss, kld_loss

def vae_loss(x, recon_x, mu, logvar, loss_fn="MSE"):
    if loss_fn == "MSE":
        return vae_loss_mse(x, recon_x, mu, logvar)
    if loss_fn == "MSE_2":
        return vae_loss_mse2(x, recon_x, mu, logvar)
    elif loss_fn == "BCE":
        return vae_loss_bce(x, recon_x, mu, logvar)
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

def train_vae(model,
              epochs = 30,
              lr = 3e-04,
              loss_fn = "MSE"):
    device = get_device()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    trainLoader = prepare_cifar10_training_data()
    losses = {
        "loss": [],
        "recon_loss": [],
        "kld_loss": []
    }
    for epoch in range(epochs):
        loss = None
        for batch in trainLoader:
            optimizer.zero_grad()
            x = batch[0].to(device)
            recon_x, mu, logvar = model(x)
            # calculate loss
            loss, recon_loss, kld_loss = vae_loss(
                x, 
                recon_x, 
                mu, 
                logvar, 
                loss_fn=loss_fn)
            losses["loss"].append(loss.item())
            losses["recon_loss"].append(recon_loss.mean().item())
            losses["kld_loss"].append(kld_loss.mean().item())
            loss.backward()
            optimizer.step()
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Loss: {loss.item():.4f}, "
            f"Recon Loss: {recon_loss.mean().item():.4f}, "
            f"KLD Loss: {kld_loss.mean().item():.4f}"
        )
    return model, losses

def draw_loss_curve(losses):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(losses["loss"])
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss')
    plt.title('VAE Total Loss Curve')
    plt.subplot(1, 3, 2)
    plt.plot(losses["recon_loss"])
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Loss')
    plt.title('VAE Reconstruction Loss Curve')
    plt.subplot(1, 3, 3)
    plt.plot(losses["kld_loss"])
    plt.xlabel('Iteration')
    plt.ylabel('KLD Loss')
    plt.title('VAE KLD Loss Curve')
    plt.show()


def sample_vae(model):
    testLoader = prepare_cifar10_test_data()
    with torch.no_grad():
        for batch in testLoader:
            x = batch[0].to("cpu")
            recon_x, _ ,_ = model(x)

            # compare x and recon_x by showing images
            x = x.clamp(0,1).cpu().numpy().transpose(0, 2, 3, 1)
            recon_x = recon_x.clamp(0,1).cpu().numpy().transpose(0, 2, 3, 1)

            x = (x * 255).astype('uint8')
            recon_x = (recon_x * 255).astype('uint8')
            # show original and reconstructed images
            n = 8
            images = []
            titles = []
            for i in range(n):
                images.append(x[i])
                titles.append('Original')
                images.append(recon_x[i])
                titles.append('Reconstructed')
            show_images(images, titles, cols=4, figsize=(12, 6))
            break

def sample_vae2(model, count=32, sigma=0.1, temperature=1.0, loss_fn="MSE"):
    # decode without tracking gradients and return the tensor
    with torch.no_grad():
         z = torch.randn(count, model.latent_dim, device="cpu") * temperature
         if loss_fn == "BCE":
            decode_bernoulli_sample(model, z)
         else:
            decode_gaussian_sample(model, z, sigma=sigma)

def decode_gaussian_sample(model, z, sigma=0.1):
    model.eval()
    with torch.no_grad():
        proj = model.projection(z).view(
            z.size(0), model.feature_channels, model.feature_size, model.feature_size
        )
        mean = model.decode(proj)                        # in [0,1] (Sigmoid in decoder)
        sample = (mean + sigma * torch.randn_like(mean)).clamp(0, 1)
        sample, mean
    
    # both tensors expected in [0,1], shape (B, 3, 32, 32)
    grid_sample = torchvision.utils.make_grid(sample, nrow=8)
    grid_mean   = torchvision.utils.make_grid(mean,   nrow=8)

    # Save to disk
    fname_prefix = f"vae_gaussian_sigma{sigma}"
    torchvision.utils.save_image(grid_sample, f"{fname_prefix}_sample.png")
    torchvision.utils.save_image(grid_mean,   f"{fname_prefix}_mean.png")

    print(f"wrote {fname_prefix}_sample.png and {fname_prefix}_mean.png")
    
def decode_bernoulli_sample(model, z):
    model.eval()
    with torch.no_grad():
        proj = model.projection(z).view(z.size(0), model.feature_channels, model.feature_size, model.feature_size)
        probs = model.decode(proj).clamp(0, 1)      # raw logits

    grid_p = torchvision.utils.make_grid(probs, nrow=8)
    torchvision.utils.save_image(grid_p, "vae_bernoulli_mean.png")
    print("Saved: vae_bernoulli_mean.png")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    trainLoader = prepare_cifar10_training_data()
    batch = next(iter(trainLoader))
    sample = batch[0] #[64, 3, 32, 32]
    print("training sample: ", sample.shape)
    img = sample[0]  #[1, 3, 32, 32]
    print(img.shape)
    print(img.min().item(), img.max().item())

    # model definition
    in_channels = 3
    feature_size = 2
    feature_channels = 512
    latent_dim = 256
    loss_fn = "BCE"
    model = VAE(in_channels=in_channels,
                latent_dim=latent_dim,
                feature_size=feature_size,
                feature_channels=feature_channels)
    # print the model architecture
    torchsummary.summary(model, input_size=(3, 32, 32))

    model_file = f'vae_cifar10.pth'
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
        print("Loaded pre-trained model.")
    else:
        epochs = 50
        lr = 3e-04
        model, losses = train_vae(
            model,
            epochs=epochs,
            lr=lr,
            loss_fn=loss_fn
        )
        draw_loss_curve(losses)
        
        # save all hyperparameters in a separate json file
        hyperparams = {
            "epochs": epochs,
            "lr": lr,
            "latent_dim": latent_dim,
            "feature_size": feature_size,
            "feature_channels": feature_channels,
            "loss_fn": loss_fn
        }
        with open(model_file.replace('.pth', '.json'), 'w') as f:
            json.dump(hyperparams, f)

        # save the trained model
        torch.save(model.state_dict(), model_file)

    # Model Inference
    model.to('cpu')
    model.eval()
    sample_vae(model)
    sample_vae2(model, 8, loss_fn=loss_fn)



if __name__ == "__main__":
    main()