import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    """
    Build a DataLoader for aligned CelebA images.

    CelebA's aligned images are 178x218. Center-cropping to 178x178 removes
    the extra vertical area while preserving the dataset's face alignment.
    Normalizing with mean=std=0.5 maps ToTensor's [0, 1] RGB values to
    [-1, 1], which matches the range expected by the diffusion model.

    Example arguments:

    args.image_size = 64
    args.batch_size = 32
    args.dataset_path = "data"
    args.download = True
    args.num_workers = 4
    args.split = "train"
    args.max_train_samples = 20_000
    args.subset_seed = 42
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(178),
            torchvision.transforms.Resize(
                (args.image_size, args.image_size),
                antialias=True,
            ),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    # CelebA is a flat collection of aligned images plus metadata, not a set
    # of class directories, so ImageFolder is not the appropriate dataset.
    dataset = torchvision.datasets.CelebA(
        root=args.dataset_path,
        split=getattr(args, "split", "train"),
        # Keep a collatable target even though unconditional training ignores
        # it. An empty target list makes CelebA return None for every sample,
        # which the default DataLoader collate function cannot batch.
        target_type="attr",
        transform=transform,
        download=getattr(args, "download", False),
    )

    # Use a deterministic random subset for quicker DDPM experiments. Taking
    # the first N filenames could preserve ordering biases in the source data,
    # while a fixed generator seed gives the same representative subset on
    # every run. Set max_train_samples=0 to use the entire training split.
    max_train_samples = getattr(args, "max_train_samples", 0)
    if max_train_samples < 0:
        raise ValueError("max_train_samples must be non-negative")

    split = getattr(args, "split", "train")
    if split == "train" and max_train_samples > 0:
        subset_size = min(max_train_samples, len(dataset))
        generator = torch.Generator().manual_seed(getattr(args, "subset_seed", 42))
        indices = torch.randperm(
            len(dataset),
            generator=generator,
        )[:subset_size].tolist()
        dataset = Subset(dataset, indices)

    num_workers = getattr(args, "num_workers", 0)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
