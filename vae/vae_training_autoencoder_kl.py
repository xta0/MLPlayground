import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from vae_training import SUPPORTED_DATASETS, prepare_image_data


def group_count(channels, preferred_groups=32):
    for groups in range(min(preferred_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiagonalGaussianDistribution:
    def __init__(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        self.mean = mean
        self.logvar = logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.std)

    def mode(self):
        return self.mean

    def kl(self):
        kl = 0.5 * (
            self.mean.pow(2) + self.logvar.exp() - 1.0 - self.logvar
        )
        return kl.flatten(1).sum(dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + residual


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class AutoencoderKLEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, latent_channels=4, dropout=0.0):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down = nn.Sequential(
            ResBlock(channels[0], channels[0], dropout=dropout),
            ResBlock(channels[0], channels[0], dropout=dropout),
            Downsample(channels[0]),
            ResBlock(channels[0], channels[1], dropout=dropout),
            ResBlock(channels[1], channels[1], dropout=dropout),
            Downsample(channels[1]),
            ResBlock(channels[1], channels[2], dropout=dropout),
            ResBlock(channels[2], channels[2], dropout=dropout),
            Downsample(channels[2]),
        )
        self.mid = nn.Sequential(
            ResBlock(channels[2], channels[2], dropout=dropout),
            ResBlock(channels[2], channels[2], dropout=dropout),
        )
        self.norm_out = nn.GroupNorm(group_count(channels[2]), channels[2])
        self.quant_conv = nn.Conv2d(channels[2], latent_channels * 2, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down(x)
        x = self.mid(x)
        x = F.silu(self.norm_out(x))
        return self.quant_conv(x)


class AutoencoderKLDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=128, latent_channels=4, dropout=0.0):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.post_quant_conv = nn.Conv2d(latent_channels, channels[2], kernel_size=1)
        self.mid = nn.Sequential(
            ResBlock(channels[2], channels[2], dropout=dropout),
            ResBlock(channels[2], channels[2], dropout=dropout),
        )
        self.up = nn.Sequential(
            ResBlock(channels[2], channels[2], dropout=dropout),
            Upsample(channels[2]),
            ResBlock(channels[2], channels[1], dropout=dropout),
            ResBlock(channels[1], channels[1], dropout=dropout),
            Upsample(channels[1]),
            ResBlock(channels[1], channels[0], dropout=dropout),
            ResBlock(channels[0], channels[0], dropout=dropout),
            Upsample(channels[0]),
            ResBlock(channels[0], channels[0], dropout=dropout),
        )
        self.norm_out = nn.GroupNorm(group_count(channels[0]), channels[0])
        self.conv_out = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        z = self.post_quant_conv(z)
        z = self.mid(z)
        z = self.up(z)
        z = self.conv_out(F.silu(self.norm_out(z)))
        return torch.tanh(z)


class AutoencoderKL(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, latent_channels=4, dropout=0.0):
        super().__init__()
        self.encoder = AutoencoderKLEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            dropout=dropout,
        )
        self.decoder = AutoencoderKLDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            dropout=dropout,
        )
        self.latent_channels = latent_channels

    def encode(self, x):
        moments = self.encoder(x)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        recon_x = self.decode(z)
        return recon_x, posterior


def autoencoder_kl_loss(x, recon_x, posterior, kl_weight=1e-4, mse_weight=0.25):
    l1_per_sample = F.l1_loss(recon_x, x, reduction="none").mean(dim=(1, 2, 3))
    mse_per_sample = F.mse_loss(recon_x, x, reduction="none").mean(dim=(1, 2, 3))
    recon_per_sample = l1_per_sample + mse_weight * mse_per_sample
    kl_per_sample = posterior.kl()
    loss = (recon_per_sample + kl_weight * kl_per_sample).mean()
    return loss, recon_per_sample, kl_per_sample


def checkpoint_path_for_epoch(model_file, epoch):
    stem, ext = os.path.splitext(model_file)
    return f"{stem}_epoch_{epoch:03d}{ext or '.pth'}"


def train_autoencoder(model,
                      train_loader,
                      epochs=50,
                      lr=3e-4,
                      kl_weight=1e-4,
                      mse_weight=0.25,
                      checkpoint_every=10,
                      model_file=None):
    device = get_device()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.9), eps=1e-8)
    losses = {"loss": [], "recon_loss": [], "kld_loss": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        batches = 0

        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon_x, posterior = model(x)
            loss, recon_loss, kld_loss = autoencoder_kl_loss(
                x,
                recon_x,
                posterior,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
            )
            loss.backward()
            optimizer.step()

            losses["loss"].append(loss.item())
            losses["recon_loss"].append(recon_loss.mean().item())
            losses["kld_loss"].append(kld_loss.mean().item())
            epoch_loss += loss.item()
            epoch_recon += recon_loss.mean().item()
            epoch_kl += kld_loss.mean().item()
            batches += 1

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {epoch_loss / batches:.4f}, "
            f"Recon Loss: {epoch_recon / batches:.4f}, "
            f"KLD Loss: {epoch_kl / batches:.4f}"
        )
        if model_file is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_file = checkpoint_path_for_epoch(model_file, epoch + 1)
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Saved checkpoint: {checkpoint_file}")

    return model, losses


def draw_loss_curve(losses):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(losses["loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("AutoencoderKL Total Loss")
    plt.subplot(1, 3, 2)
    plt.plot(losses["recon_loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Reconstruction Loss")
    plt.title("Hybrid Reconstruction Loss")
    plt.subplot(1, 3, 3)
    plt.plot(losses["kld_loss"])
    plt.xlabel("Iteration")
    plt.ylabel("KLD Loss")
    plt.title("Spatial KL Loss")
    plt.show()


def load_state_dict(model_file):
    try:
        return torch.load(model_file, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(model_file, map_location="cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="celeba64")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_size % 8 != 0:
        raise ValueError("image_size must be divisible by 8 for the AutoencoderKL spatial latent VAE.")

    train_loader = prepare_image_data(
        dataset_name=args.dataset,
        split="train",
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        normalize=True,
        shuffle=True,
        num_workers=args.num_workers,
        download=args.download,
        subset_fraction=args.train_fraction,
        subset_seed=args.subset_seed,
    )

    batch = next(iter(train_loader))
    sample = batch[0]
    print("training sample:", sample.shape)
    print("value range:", sample.min().item(), sample.max().item())

    model = AutoencoderKL(
        in_channels=3,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        dropout=args.dropout,
    )

    latent_size = args.image_size // 8
    print(
        "spatial latent shape:",
        (args.batch_size, args.latent_channels, latent_size, latent_size),
    )

    model_file = args.model_file or f"vae_autoencoder_kl_{args.dataset}_{args.image_size}.pth"
    if os.path.exists(model_file):
        model.load_state_dict(load_state_dict(model_file))
        print("Loaded pre-trained AutoencoderKL model.")
    else:
        model, losses = train_autoencoder(
            model,
            train_loader,
            epochs=args.epochs,
            lr=args.lr,
            kl_weight=args.kl_weight,
            mse_weight=args.mse_weight,
            checkpoint_every=args.checkpoint_every,
            model_file=model_file,
        )
        draw_loss_curve(losses)

        hyperparams = {
            "dataset": args.dataset,
            "data_root": args.data_root,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "base_channels": args.base_channels,
            "latent_channels": args.latent_channels,
            "latent_size": latent_size,
            "dropout": args.dropout,
            "kl_weight": args.kl_weight,
            "mse_weight": args.mse_weight,
            "train_fraction": args.train_fraction,
            "subset_seed": args.subset_seed,
            "checkpoint_every": args.checkpoint_every,
            "input_range": "[-1, 1]",
            "output_activation": "tanh",
        }
        with open(model_file.replace(".pth", ".json"), "w") as f:
            json.dump(hyperparams, f)

        torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    main()
