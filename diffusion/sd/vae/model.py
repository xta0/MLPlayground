import torch
from torch import nn
from torch.nn import functional as F


def group_count(channels, preferred_groups=32):
    for groups in range(min(preferred_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


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
        # Average over channels and spatial positions so the KL scale does
        # not change when the latent resolution changes.
        return kl.mean(dim=(1, 2, 3))


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
    def __init__(
        self,
        in_channels=3,
        base_channels=128,
        latent_channels=4,
        dropout=0.0,
    ):
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
    def __init__(
        self,
        out_channels=3,
        base_channels=128,
        latent_channels=4,
        dropout=0.0,
    ):
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
    def __init__(
        self,
        in_channels=3,
        base_channels=128,
        latent_channels=4,
        dropout=0.0,
    ):
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
        self.downsample_factor = 4

    def encode(self, x):
        height, width = x.shape[-2:]
        if (
            height % self.downsample_factor != 0
            or width % self.downsample_factor != 0
        ):
            raise ValueError(
                "Input height and width must be divisible by "
                f"{self.downsample_factor}; received {height}x{width}."
            )
        moments = self.encoder(x)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        recon_x = self.decode(z)
        return recon_x, posterior
