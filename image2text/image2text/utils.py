
import torch
from torch import nn
import math

def extract_patches(image_tenosr, patch_size=16):
    """
    Given an input of shape (N, C, H, W), the output will be a tensor of shape (N, C *
    kernel_height * kernel_width, L), where L is the total number of sliding blocks. If we
    apply the Unfold() layer on one image with an input shape of (1, 3, 128, 128), the out-
    put has a shape of (1, 768, 64): one image in the batch, each image patch has 3 x 16 x 16
    = 768 pixels, and there are a total of 64 sliding blocks.
    """
    B, C, H, W = image_tenosr.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    patches = unfold(image_tenosr) # [B, C*patch_size*patch_size, num_patches]
    # [B, num_patches, pixel_per_patch] -> [1, 64, 768]
    patches = patches.transpose(1, 2).reshape(B, -1, C*patch_size*patch_size)
    return patches

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
