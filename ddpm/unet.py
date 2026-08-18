import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
                    residual (optional)
                  ┌─────────────────────┐
    x ────────────┤                     + → GELU → output
                  └→ Conv → Norm → GELU → Conv → Norm ┘
    """

    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()

        if residual and in_channels != out_channels:
            raise ValueError("Residual DoubleConv requires in_channels == out_channels")

        if out_channels % 8 != 0:
            raise ValueError("out_channels must be divisible by 8 for GroupNorm")

        self.residual = residual

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
        )

    def forward(self, x):
        output = self.double_conv(x)
        if self.residual:
            return F.gelu(x + output)
        return output


class Down(nn.Module):
    """
    Feature map
      ↓
    MaxPool: halve height and width
      ↓
    Residual DoubleConv: refine features
      ↓
    DoubleConv: change channel count
      ↓
    Add timestep embedding
    """

    def __init__(self, in_channels, out_channels, embed_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # Channels stay equal so residual addition is valid.
            DoubleConv(in_channels, in_channels, residual=True),
            # Change channels after the residual block.
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, out_channels))

    def forward(self, x, t):
        """
        t here should already be the sinusoidal timestep encoding shaped [B, 256],
        not the original integer timestep tensor shaped [B].

        The usual UNet flow is:
        t = t.unsqueeze(-1).float()       # [B, 1]
        t = self.pos_encoding(t, 256)     # [B, 256]

        For down1 = Down(64, 128), the intended shapes are:

        x:                   [B,  64, 64, 64]
        MaxPool2d(2):         [B,  64, 32, 32]
        Residual DoubleConv:  [B,  64, 32, 32]
        Channel DoubleConv:   [B, 128, 32, 32]
        Timestep embedding:   [B, 128,  1,  1]
        Output:               [B, 128, 32, 32]
        """
        x = self.maxpool_conv(x)  # [B, out_channels, H/2, W/2]

        emb = self.emb_layer(t).view(
            t.shape[0], -1, 1, 1
        )  # [B, out_channels, 1, 1] - Broadcasting over height and width
        return x + emb


class Up(nn.Module):
    """
    Feature map
      ↓
    Upsample: double height and width
      ↓
    Residual DoubleConv: refine features
      ↓
    DoubleConv: change channel count
      ↓
    Add timestep embedding
    """

    def __init__(self, in_channels, out_channels, embed_dim=256):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        self.conv = nn.Sequential(
            # Channels stay equal so residual addition is valid.
            DoubleConv(in_channels, in_channels, residual=True),
            # Change channels after the residual block.
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, out_channels))

    def forward(self, x, skip_x, t):
        """
        t here should already be the sinusoidal timestep encoding shaped [B, 256],
        not the original integer timestep tensor shaped [B].

        The usual UNet flow is:
        t = t.unsqueeze(-1).float()       # [B, 1]
        t = self.pos_encoding(t, 256)     # [B, 256]

        For up1 = Up(512, 128), the intended shapes are:

        x:                   [B, 512, 16, 16]
        Upsample:             [B, 512, 32, 32]
        Residual DoubleConv:  [B, 512, 32, 32]
        Channel DoubleConv:   [B, 128, 32, 32]
        Timestep embedding:   [B, 128,  1,  1]
        Output:               [B, 128, 32, 32]
        """
        x = self.upsample(x)  # [B, out_channels, H*2, W*2]
        # skip connection
        x = torch.cat([skip_x, x], dim=1)
        # conv
        x = self.conv(x)  # [B, out_channels, H*2, W*2]

        emb = self.emb_layer(t).view(
            t.shape[0], -1, 1, 1
        )  # [B, out_channels, 1, 1] - Broadcasting over height and width
        return x + emb


class SelfAttention(nn.Module):
    """
    Self-Attention block for 2D feature maps.

    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.channels = channels
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(channels)
        """
        MultiheadAttention expects input of shape (B, S, E) where:
        - B is the batch size
        - S is the sequence length
        - E is the embedding dimension
        """
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # Pre-normalized attention with identity residual.
        normalized = self.norm1(tokens)

        # attention
        attn, _ = self.mha(
            normalized, normalized, normalized, need_weights=False
        )  # (B, H*W, C)

        # residual connection
        tokens = tokens + attn

        tokens = tokens + self.ff(self.norm2(tokens))  # (B, H*W, C)

        # [B, H*W, C] → [B, C, H, W]
        output = tokens.transpose(1, 2).reshape(
            B,
            C,
            H,
            W,
        )

        return output


class UNet(nn.Module):
    """
    encoder -> bottleneck -> decoder
    """

    def __init__(self, c_in=3, c_out=3, time_dim=256):
        super().__init__()

        # The sinusoidal encoding uses half of the dimensions for sine and
        # half for cosine, so an odd embedding size is not supported.
        if time_dim % 2 != 0:
            raise ValueError("time_dim must be even")

        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, embed_dim=time_dim)

        # Omit global attention at this high-resolution encoder level. For a
        # 128x128 input this feature map is 64x64, or 4096 spatial tokens.
        self.sa1 = nn.Identity()

        self.down2 = Down(128, 256, embed_dim=time_dim)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256, embed_dim=time_dim)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, embed_dim=time_dim)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64, embed_dim=time_dim)

        # This is the matching high-resolution decoder level, so skip global
        # attention here for the same reason as sa1.
        self.sa5 = nn.Identity()

        self.up3 = Up(128, 64, embed_dim=time_dim)

        # Full output-resolution attention would be even more expensive:
        # 4096 tokens at 64x64 and 16384 tokens at 128x128.
        self.sa6 = nn.Identity()
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # Three 2x downsampling stages require dimensions divisible by 2**3.
        # Otherwise upsampled decoder features will not match encoder skips.
        height, width = x.shape[-2:]
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "Input height and width must both be divisible by 8; "
                f"received {height}x{width}"
            )

        t = t.unsqueeze(-1).float()  # [B, 1]
        t = self.pos_encoding(t, self.time_dim)  # [B, 256]

        # encoder path
        x1 = self.inc(x)  # [B, 64, 64, 64]
        x2 = self.down1(x1, t)  # [B, 128, 32, 32]
        x2 = self.sa1(x2)  # [B, 128, 32, 32]
        x3 = self.down2(x2, t)  # [B, 256, 16, 16]
        x3 = self.sa2(x3)  # [B, 256, 16, 16]
        x4 = self.down3(x3, t)  # [B, 256, 8, 8]
        x4 = self.sa3(x4)  # [B, 256, 8, 8]

        # bottleneck
        x4 = self.bot1(x4)  # [B, 512, 8, 8]
        x4 = self.bot2(x4)  # [B, 512, 8, 8]
        x4 = self.bot3(x4)  # [B, 256, 8, 8]

        # decoder path
        """
        skip connection:

        16x16 encoder x3 ─────→ up1
        32x32 encoder x2 ─────→ up2
        64x64 encoder x1 ─────→ up3
        """
        x = self.up1(x4, x3, t)  # [B, 128, 16, 16]
        x = self.sa4(x)  # [B, 128, 16, 16]
        x = self.up2(x, x2, t)  # [B, 64, 32, 32]
        x = self.sa5(x)  # [B, 64, 32, 32] (identity)
        x = self.up3(x, x1, t)  # [B, 64, 64, 64]
        x = self.sa6(x)  # [B, 64, 64, 64] (identity)
        output = self.outc(x)  # [B, 3, 64, 64]
        return output
