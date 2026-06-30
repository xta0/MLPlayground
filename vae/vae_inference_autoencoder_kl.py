import argparse
import json
import os
import re

import torch
import torchvision

from vae_training import SUPPORTED_DATASETS, prepare_image_data
from vae_training_autoencoder_kl import AutoencoderKL, get_device


def load_training_config(model_file):
    stem, _ = os.path.splitext(model_file)
    candidate_files = [f"{stem}.json"]
    base_stem = re.sub(r"_epoch_\d+$", "", stem)
    if base_stem != stem:
        candidate_files.append(f"{base_stem}.json")

    for config_file in candidate_files:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                return json.load(f)
    return {}


def resolve_settings(args):
    dataset = args.dataset or "celeba64"
    image_size = args.image_size or 64
    model_file = args.model_file or f"vae_autoencoder_kl_{dataset}_{image_size}.pth"
    config = load_training_config(model_file)

    dataset = args.dataset or config.get("dataset", dataset)
    image_size = args.image_size or config.get("image_size", image_size)

    return {
        "model_file": model_file,
        "dataset": dataset,
        "data_root": args.data_root or config.get("data_root", "./data"),
        "image_size": image_size,
        "batch_size": args.batch_size or config.get("batch_size", 64),
        "num_workers": args.num_workers if args.num_workers is not None else 2,
        "base_channels": args.base_channels or config.get("base_channels", 128),
        "latent_channels": args.latent_channels or config.get("latent_channels", 4),
        "dropout": config.get("dropout", 0.0),
    }


def load_model(settings, device):
    model_file = settings["model_file"]
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = AutoencoderKL(
        in_channels=3,
        base_channels=settings["base_channels"],
        latent_channels=settings["latent_channels"],
        dropout=settings["dropout"],
    )
    try:
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_file, map_location="cpu")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def to_image_range(images):
    return ((images + 1.0) / 2.0).clamp(0, 1)


def show_reconstructions(model, data_loader, count=8, output_path=".", nrow=4):
    device = next(model.parameters()).device
    os.makedirs(output_path, exist_ok=True)
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0][:count].to(device)
            recon_x, _ = model(x)

            images = torch.stack(
                (to_image_range(x).cpu(), to_image_range(recon_x).cpu()),
                dim=1
            ).flatten(0, 1)
            reconstruction_path = os.path.join(output_path, "reconstructions.png")
            torchvision.utils.save_image(images, reconstruction_path, nrow=nrow)
            print(f"wrote {reconstruction_path}")
            break


def generate_vae_samples(model, image_size, count=32, temperature=1.0, output_path=".", nrow=8):
    device = next(model.parameters()).device
    latent_size = image_size // 8
    model.eval()
    with torch.no_grad():
        z = torch.randn(
            count,
            model.latent_channels,
            latent_size,
            latent_size,
            device=device,
        ) * temperature
        images = model.decode(z)

    images = to_image_range(images).cpu()
    os.makedirs(output_path, exist_ok=True)

    for index, image in enumerate(images):
        image_path = os.path.join(output_path, f"sample_{index:03d}.png")
        torchvision.utils.save_image(image, image_path)
        print(f"wrote {image_path}")

    grid_path = os.path.join(output_path, "generated_grid.png")
    torchvision.utils.save_image(images, grid_path, nrow=nrow)
    print(f"wrote {grid_path}")
    return images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-path", default=".")
    parser.add_argument("--reconstruction-count", type=int, default=8)
    parser.add_argument("--nrow", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    settings = resolve_settings(args)
    device = get_device()
    model = load_model(settings, device)

    sample_loader = prepare_image_data(
        dataset_name=settings["dataset"],
        split="test",
        data_root=settings["data_root"],
        image_size=settings["image_size"],
        batch_size=settings["batch_size"],
        normalize=True,
        shuffle=False,
        num_workers=settings["num_workers"],
        download=False,
    )

    show_reconstructions(
        model,
        sample_loader,
        count=args.reconstruction_count,
        output_path=args.output_path,
    )

    generate_vae_samples(
        model,
        image_size=settings["image_size"],
        count=args.sample_count,
        temperature=args.temperature,
        output_path=args.output_path,
        nrow=args.nrow,
    )


if __name__ == "__main__":
    main()
