import argparse
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw

from ddpm import Diffusion
from unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare real CelebA images with samples from a DDPM checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/DDPM-full-ema/ckpt.pt",
    )
    parser.add_argument("--dataset-path", default="data")
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="train",
        help="Official CelebA split from which to draw the real images.",
    )
    parser.add_argument(
        "--output",
        default="results/DDPM-full-ema/real-vs-generated.png",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    return parser.parse_args()


def load_real_images(dataset_path, split, image_size, num_samples, seed):
    # Use the same deterministic crop and resize as training, but omit random
    # flipping and [-1, 1] normalization so the grid shows natural RGB images.
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(178),
            torchvision.transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            torchvision.transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.CelebA(
        root=dataset_path,
        split=split,
        target_type="attr",
        transform=transform,
        download=False,
    )

    # A private generator makes real-image selection reproducible without
    # consuming randomness from the diffusion sampler.
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples]
    return torch.stack([dataset[index.item()][0] for index in indices])


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Prefer the smoothed EMA weights. Fall back to the training weights so
    # this utility also works with checkpoints created before EMA support.
    weights_key = "ema_model" if "ema_model" in checkpoint else "model"
    model = UNet()
    model.load_state_dict(checkpoint[weights_key])
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint, weights_key


def tensor_grid_to_image(images, columns):
    grid = torchvision.utils.make_grid(
        images,
        nrow=columns,
        padding=2,
    )
    pixels = grid.clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels, mode="RGB")


def combine_grids(real_grid, generated_grid, split):
    header_height = 28
    section_gap = 8
    width = max(real_grid.width, generated_grid.width)
    height = (
        header_height
        + real_grid.height
        + section_gap
        + header_height
        + generated_grid.height
    )

    comparison = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(comparison)
    draw.text((8, 8), f"Real CelebA {split} images", fill="black")
    comparison.paste(real_grid, (0, header_height))

    generated_header_y = header_height + real_grid.height + section_gap
    draw.text((8, generated_header_y + 8), "Generated with EMA model", fill="black")
    comparison.paste(generated_grid, (0, generated_header_y + header_height))
    return comparison


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if args.columns <= 0:
        raise ValueError("columns must be positive")

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    real_images = load_real_images(
        args.dataset_path,
        args.split,
        args.image_size,
        args.num_samples,
        args.seed,
    )

    device = torch.device(args.device)
    model, checkpoint, weights_key = load_model(checkpoint_path, device)
    diffusion = Diffusion(img_size=args.image_size, device=device)

    torch.manual_seed(args.seed)
    generated_images, _ = diffusion.sample(model, args.num_samples)
    generated_images = generated_images.float().div(255)

    real_grid = tensor_grid_to_image(real_images, args.columns)
    generated_grid = tensor_grid_to_image(generated_images, args.columns)
    comparison = combine_grids(real_grid, generated_grid, args.split)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path)
    print(f"Loaded {weights_key} weights from epoch {checkpoint['epoch']}")
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
