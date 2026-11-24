from dataclasses import dataclass
from pathlib import Path

BEST_CHECKPOINT = Path("checkpoints/best.pth")

@dataclass
class Config:
    image_size: int = 128
    channels: int = 3
    patch_size: int = 8
    hidden_size: int = 192
    num_attention_heads: int = 8
    encoder_transformer_blocks: int = 6
    decoder_transformer_blocks: int = 6
