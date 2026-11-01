import argparse
import json
import os

import torch
import torchvision

from .data import SUPPORTED_DATASETS, prepare_image_data
from .model import AutoencoderKL
from .utils import get_device, load_torch_file, model_state_dict


def load_config(checkpoint_file, payload):
    if isinstance(payload, dict) and "config" in payload:
        return payload["config"]

    config_file = os.path.splitext(checkpoint_file)[0] + ".json"
    if not os.path.exists(config_file):
        return {}
    with open(config_file) as file:
        return json.load(file)


def save_reconstructions(model, data_loader, output_file, count=8):
    device = next(model.parameters()).device
    with torch.no_grad():
        images = next(iter(data_loader))[0][:count].to(device)
        reconstructions, _ = model(images, sample_posterior=False)

    pairs = torch.stack(
        (
            ((images + 1.0) / 2.0).clamp(0, 1).cpu(),
            ((reconstructions + 1.0) / 2.0).clamp(0, 1).cpu(),
        ),
        dim=1,
    ).flatten(0, 1)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torchvision.utils.save_image(pairs, output_file, nrow=4)
    print(f"Saved reconstructions to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        "--model-file",
        dest="checkpoint",
        required=True,
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--output-file",
        default="results/vae/reconstructions.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = load_torch_file(args.checkpoint)
    config = load_config(args.checkpoint, payload)
    dataset = args.dataset or config.get("dataset", "celeba64")
    data_root = args.data_root or config.get("data_root", "./data")
    image_size = args.image_size or config.get("image_size", 128)
    base_channels = args.base_channels or config.get("base_channels", 128)
    latent_channels = args.latent_channels or config.get("latent_channels", 4)
    dropout = config.get("dropout", 0.0)

    device = get_device()
    model = AutoencoderKL(
        in_channels=3,
        base_channels=base_channels,
        latent_channels=latent_channels,
        dropout=dropout,
    )
    model.load_state_dict(model_state_dict(payload))
    model.to(device).eval()

    data_loader = prepare_image_data(
        dataset_name=dataset,
        split="test",
        data_root=data_root,
        image_size=image_size,
        batch_size=args.batch_size,
        normalize=True,
        shuffle=False,
        num_workers=args.num_workers,
    )
    save_reconstructions(
        model,
        data_loader,
        args.output_file,
        count=args.count,
    )


if __name__ == "__main__":
    main()
