import os
import json
import argparse
import torch
import torchsummary
from torch import nn
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

SUPPORTED_DATASETS = ("celeba64", "imagefolder64")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FlatImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None, train_fraction=0.95):
        self.root = root
        self.transform = transform
        all_paths = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    all_paths.append(os.path.join(dirpath, filename))

        all_paths.sort()
        if not all_paths:
            raise ValueError(f"No images found under {root}.")

        if split == "all":
            self.paths = all_paths
        elif split == "train":
            split_index = max(1, int(len(all_paths) * train_fraction))
            self.paths = all_paths[:split_index]
        else:
            split_index = max(1, int(len(all_paths) * train_fraction))
            self.paths = all_paths[split_index:] or all_paths[-1:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def has_images(path):
    if not os.path.isdir(path):
        return False
    return any(
        filename.lower().endswith(IMAGE_EXTENSIONS)
        for filename in os.listdir(path)
    )


def resolve_celeba_image_root(data_root, split):
    split_root = os.path.join(data_root, split)
    if has_images(split_root):
        return split_root, "all"

    celeba_root = os.path.join(data_root, "img_align_celeba")
    if has_images(celeba_root):
        return celeba_root, split

    if has_images(data_root):
        return data_root, split

    return None, None


def build_image_transform(image_size=64, normalize=False, center_crop=None):
    transform_steps = []
    if center_crop is not None:
        transform_steps.append(transforms.CenterCrop(center_crop))
    transform_steps.extend([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), # converts the pixel to [0, 1]
    ])
    if normalize:
        transform_steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_steps)


def prepare_image_data(dataset_name="celeba64",
                       split="train",
                       data_root="./data",
                       image_size=64,
                       batch_size=64,
                       normalize=False,
                       shuffle=True,
                       num_workers=2,
                       download=False,
                       subset_fraction=1.0,
                       subset_seed=42):
    if image_size % 16 != 0:
        raise ValueError("image_size must be divisible by 16 for the current encoder/decoder.")

    if dataset_name == "celeba64":
        transform = build_image_transform(
            image_size=image_size,
            normalize=normalize,
            center_crop=178
        )
        try:
            dataset = datasets.CelebA(
                root=data_root,
                split=split,
                target_type="attr",
                download=download,
                transform=transform
            )
        except Exception as exc:
            image_root, dataset_split = resolve_celeba_image_root(data_root, split)
            if image_root is None:
                raise RuntimeError(
                    "Could not load Torchvision CelebA. Torchvision downloads CelebA from "
                    "Google Drive, which is often quota-limited. Either retry later with "
                    "--download, or manually download/extract the aligned CelebA images and "
                    "run with: --dataset celeba64 --data-root /path/to/img_align_celeba"
                ) from exc
            dataset = FlatImageDataset(image_root, split=dataset_split, transform=transform)
    elif dataset_name == "imagefolder64":
        transform = build_image_transform(image_size=image_size, normalize=normalize)
        split_root = os.path.join(data_root, split)
        image_root = split_root if os.path.isdir(split_root) else data_root
        dataset = datasets.ImageFolder(image_root, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose one of {SUPPORTED_DATASETS}.")

    if not 0 < subset_fraction <= 1:
        raise ValueError("subset_fraction must be greater than 0 and less than or equal to 1.")
    if subset_fraction < 1:
        subset_size = max(1, int(len(dataset) * subset_fraction))
        generator = torch.Generator().manual_seed(subset_seed)
        indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


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
    Four stride-2 convolution blocks. For 64x64 input this maps:
    64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4.
    """
    def __init__(self, in_channels, feature_channels):
        super().__init__()
        self.conv1 = conv_block(in_channels, feature_channels // 8)
        self.conv2 = conv_block(feature_channels // 8, feature_channels // 4)
        self.conv3 = conv_block(feature_channels // 4, feature_channels // 2)
        self.conv4 = conv_block(feature_channels // 2, feature_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
    
class VAEDecoder(nn.Module):
    """
    Four upsample blocks mirroring the encoder. For 64x64 training,
    feature_size should be 4 so output returns to 64x64.
    """
    def __init__(self, feature_channels, out_channels):
        super().__init__()
        self.up1 = upsample_block(feature_channels, feature_channels // 2)
        self.up2 = upsample_block(feature_channels // 2, feature_channels // 4)
        self.up3 = upsample_block(feature_channels // 4, feature_channels // 8)
        self.up4 = upsample_block(feature_channels // 8, feature_channels // 16)
        self.conv_out = nn.Conv2d(feature_channels // 16, out_channels, kernel_size=3, padding=1, bias=True)
        self.out = nn.Sigmoid() # ensure output is in [0, 1]

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.conv_out(x)
        x = self.out(x)
        return x

class VAE(nn.Module):
    def __init__(self, in_channels=3, 
                 latent_dim=256, 
                 feature_size=4,
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
        # x comes in as (B, Cfeat, feature_size, feature_size) from encoder
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

def vae_loss_mse(x, recon_x, mu, logvar, beta=1e-3):
    # per-image mean MSE (size-invariant)
    per_img_mse = torch.nn.functional.mse_loss(
        recon_x, x, reduction='none'
    ).mean(dim=(1,2,3))                           # [B]

    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)                                             # [B]

    per_sample = per_img_mse + beta * kld_per_sample
    return per_sample.mean(), per_img_mse, kld_per_sample

def vae_loss(x, recon_x, mu, logvar):
    return vae_loss_mse(x, recon_x, mu, logvar)

def train_vae(model,
              train_loader,
              epochs = 30,
              lr = 3e-04):
    device = get_device()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    losses = {
        "loss": [],
        "recon_loss": [],
        "kld_loss": []
    }
    for epoch in range(epochs):
        loss = None
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[0].to(device)
            recon_x, mu, logvar = model(x)
            # calculate loss
            loss, recon_loss, kld_loss = vae_loss(
                x, 
                recon_x, 
                mu, 
                logvar)
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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="celeba64")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-04)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--feature-channels", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_loader = prepare_image_data(
        dataset_name=args.dataset,
        split="train",
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        download=args.download,
        subset_fraction=args.train_fraction,
        subset_seed=args.subset_seed
    )

    batch = next(iter(train_loader))
    sample = batch[0]
    print("training sample: ", sample.shape)
    img = sample[0]
    print(img.shape)
    print(img.min().item(), img.max().item())

    # model definition
    in_channels = 3
    feature_size = args.image_size // 16
    model = VAE(in_channels=in_channels,
                latent_dim=args.latent_dim,
                feature_size=feature_size,
                feature_channels=args.feature_channels)
    # print the model architecture
    torchsummary.summary(model, input_size=(in_channels, args.image_size, args.image_size))

    model_file = args.model_file or f'vae_{args.dataset}_{args.image_size}.pth'
    if os.path.exists(model_file):
        try:
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict)
        print("Loaded pre-trained model.")
    else:
        model, losses = train_vae(
            model,
            train_loader,
            epochs=args.epochs,
            lr=args.lr
        )
        draw_loss_curve(losses)
        
        # save all hyperparameters in a separate json file
        hyperparams = {
            "dataset": args.dataset,
            "data_root": args.data_root,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "latent_dim": args.latent_dim,
            "feature_size": feature_size,
            "feature_channels": args.feature_channels,
        }
        with open(model_file.replace('.pth', '.json'), 'w') as f:
            json.dump(hyperparams, f)

        # save the trained model
        torch.save(model.state_dict(), model_file)



if __name__ == "__main__":
    main()
