import argparse
from pathlib import Path

import torch

from ddpm import Diffusion
from unet import UNet
from utils import save_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images from a trained DDPM checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/DDPM-full-ema/ckpt.pt",
        help="Path to the training checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="results/DDPM-full-ema/final_samples.png",
        help="Path for the generated image grid.",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # EMA weights are a smoothed version of the training weights and normally
    # produce more stable samples. Fall back to `model` for older checkpoints
    # created before EMA support was added.
    weights_key = "ema_model" if "ema_model" in checkpoint else "model"
    model = UNet()
    model.load_state_dict(checkpoint[weights_key])
    model = model.to(device)
    model.eval()

    diffusion = Diffusion(
        img_size=args.image_size,
        device=device,
    )

    torch.manual_seed(args.seed)

    images, _ = diffusion.sample(
        model,
        num_samples=args.num_samples,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_images(images, output_path, nrow=4)

    print(f"Loaded {weights_key} weights from checkpoint epoch {checkpoint['epoch']}")
    print(f"Saved {args.num_samples} samples to {output_path}")


if __name__ == "__main__":
    main()
