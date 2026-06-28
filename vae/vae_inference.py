import argparse
import json
import os

import torch
import torchvision

from vae_training import SUPPORTED_DATASETS, VAE, get_device, prepare_image_data


def load_training_config(model_file):
    config_file = model_file.replace(".pth", ".json")
    if not os.path.exists(config_file):
        return {}

    with open(config_file, "r") as f:
        return json.load(f)


def resolve_settings(args):
    dataset = args.dataset or "celeba64"
    image_size = args.image_size or 64
    model_file = args.model_file or f"vae_{dataset}_{image_size}.pth"
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
        "latent_dim": args.latent_dim or config.get("latent_dim", 256),
        "feature_channels": args.feature_channels or config.get("feature_channels", 512),
    }


def load_model(model_file, image_size, latent_dim, feature_channels, device):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = VAE(
        in_channels=3,
        latent_dim=latent_dim,
        feature_size=image_size // 16,
        feature_channels=feature_channels,
    )
    try:
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_file, map_location="cpu")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_reconstructions(model, data_loader, count=8, output_path=".", nrow=4):
    device = next(model.parameters()).device
    os.makedirs(output_path, exist_ok=True)
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0][:count].to(device)
            recon_x, _, _ = model(x)

            images = torch.stack(
                (x.clamp(0, 1).cpu(), recon_x.clamp(0, 1).cpu()),
                dim=1
            ).flatten(0, 1)
            reconstruction_path = os.path.join(output_path, "reconstructions.png")
            torchvision.utils.save_image(images, reconstruction_path, nrow=nrow)
            print(f"wrote {reconstruction_path}")
            break


def decode_mean_samples(model, z):
    with torch.no_grad():
        proj = model.projection(z).view(
            z.size(0), model.feature_channels, model.feature_size, model.feature_size
        )
        return model.decode(proj)


def generate_vae_samples(model, count=32, temperature=1.0, output_path=".", nrow=8):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z = torch.randn(count, model.latent_dim, device=device) * temperature
        images = decode_mean_samples(model, z) # [8, 3, 64, 64]

    images = images.cpu().clamp(0, 1)
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
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--feature-channels", type=int, default=None)
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
    model = load_model(
        model_file=settings["model_file"],
        image_size=settings["image_size"],
        latent_dim=settings["latent_dim"],
        feature_channels=settings["feature_channels"],
        device=device,
    )

    sample_loader = prepare_image_data(
        dataset_name=settings["dataset"],
        split="test",
        data_root=settings["data_root"],
        image_size=settings["image_size"],
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        download=False,
    )
    save_reconstructions(
        model,
        sample_loader,
        count=args.reconstruction_count,
        output_path=args.output_path,
    )

    generate_vae_samples(
        model,
        count=args.sample_count,
        temperature=args.temperature,
        output_path=args.output_path,
        nrow=args.nrow,
    )


if __name__ == "__main__":
    main()
