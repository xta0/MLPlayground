import argparse
import json
from pathlib import Path

import torch

from data import CLASSES, create_data_loader
from utils import visualize_attention_maps
from vit import ViTForClassification


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a ViT checkpoint on CIFAR-10.")
    parser.add_argument("--run", type=Path, required=True,
                        help="Checkpoint path; config.json must be in the same directory")
    parser.add_argument("--device", default=None,
                        help="Device, such as cpu or cuda:0; defaults to CUDA if available")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if not args.run.is_file():
        parser.error(f"Checkpoint not found: {args.run}")
    if not args.run.with_name("config.json").is_file():
        parser.error(f"Missing model configuration: {args.run.with_name('config.json')}")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    print("(prediction, label)")
    batch = next(iter(loader), None)
    if batch is None:
        raise ValueError("Evaluation dataset is empty")
    with torch.inference_mode():
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device.type,
                                enabled=device.type == "cuda"):
            # print("images in batch:", images.shape)
            # [32, 3, 32, 32]
            logits, attention_maps = model(images)

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        for prediction, label in zip(predictions.cpu().tolist(), labels.cpu().tolist()):
            print(f"({CLASSES[prediction]}, {CLASSES[label]})")
        total += labels.size(0)
        
    if total == 0:
        raise ValueError("Evaluation dataset is empty")

    """
    Visualize the attention maps for the first batch of images.
    """
    visualize_attention_maps(attention_maps, images, labels, predictions)
    return correct / total


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with args.run.with_name("config.json").open() as file:
        config = json.load(file)
    model = ViTForClassification(config)
    model.load_state_dict(torch.load(args.run, map_location="cpu", weights_only=True))
    model.to(device)

    loader = create_data_loader(batch_size=args.batch_size, train=False)
    print(f"Evaluating {args.run} on {device}")
    accuracy = evaluate(model, loader, device)
    print(f"First-batch accuracy: {accuracy:.4f} ({accuracy:.2%})")

if __name__ == "__main__":
    main()
