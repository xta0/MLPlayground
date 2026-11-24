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
    encoder: str = "custom"
    best_checkpoint: Path = BEST_CHECKPOINT

    @classmethod
    def for_encoder(cls, encoder):
        if encoder == "dinov2":
            return cls(encoder=encoder, image_size=224,
                       best_checkpoint=BEST_CHECKPOINT.with_name("best_dinov2.pth"))
        if encoder == "custom":
            return cls()
        raise ValueError(f"Unknown vision encoder: {encoder}")
