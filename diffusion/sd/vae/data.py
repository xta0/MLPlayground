import os

import torch
from PIL import Image
from torchvision import datasets, transforms


SUPPORTED_DATASETS = ("celeba64", "imagefolder64")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FlatImageDataset(torch.utils.data.Dataset):
    """Fallback loader for a flat directory of aligned CelebA images."""

    def __init__(self, root, split="train", transform=None, train_fraction=0.95):
        self.transform = transform
        all_paths = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    all_paths.append(os.path.join(dirpath, filename))

        all_paths.sort()
        if not all_paths:
            raise ValueError(f"No images found under {root}.")

        split_index = max(1, int(len(all_paths) * train_fraction))
        if split == "all":
            self.paths = all_paths
        elif split == "train":
            self.paths = all_paths[:split_index]
        else:
            self.paths = all_paths[split_index:] or all_paths[-1:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def _has_images(path):
    if not os.path.isdir(path):
        return False
    return any(
        filename.lower().endswith(IMAGE_EXTENSIONS)
        for filename in os.listdir(path)
    )


def _resolve_celeba_image_root(data_root, split):
    split_root = os.path.join(data_root, split)
    if _has_images(split_root):
        return split_root, "all"

    candidates = (
        os.path.join(data_root, "celeba", "img_align_celeba"),
        os.path.join(data_root, "img_align_celeba"),
    )
    for candidate in candidates:
        if _has_images(candidate):
            return candidate, split

    if _has_images(data_root):
        return data_root, split

    return None, None


def build_image_transform(
    image_size=128,
    normalize=False,
    center_crop=None,
    augment=False,
):
    transform_steps = []
    if center_crop is not None:
        transform_steps.append(transforms.CenterCrop(center_crop))
    transform_steps.append(
        transforms.Resize(
            (image_size, image_size),
            antialias=True,
        )
    )
    if augment:
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.append(transforms.ToTensor())
    if normalize:
        transform_steps.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    return transforms.Compose(transform_steps)


def prepare_image_data(
    dataset_name="celeba64",
    split="train",
    data_root="./data",
    image_size=128,
    batch_size=64,
    normalize=False,
    shuffle=True,
    num_workers=2,
    download=False,
    subset_fraction=1.0,
    subset_seed=42,
    augment=None,
):
    if image_size % 4 != 0:
        raise ValueError(
            "image_size must be divisible by 4 for the current encoder/decoder."
        )

    if dataset_name == "celeba64":
        if augment is None:
            augment = split == "train"
        transform = build_image_transform(
            image_size=image_size,
            normalize=normalize,
            center_crop=178,
            augment=augment,
        )
        try:
            dataset = datasets.CelebA(
                root=data_root,
                split=split,
                target_type="attr",
                download=download,
                transform=transform,
            )
        except Exception as exc:
            image_root, dataset_split = _resolve_celeba_image_root(data_root, split)
            if image_root is None:
                raise RuntimeError(
                    "Could not load torchvision CelebA. Either use --download or "
                    "place the aligned images under "
                    "<data-root>/celeba/img_align_celeba."
                ) from exc
            dataset = FlatImageDataset(
                image_root,
                split=dataset_split,
                transform=transform,
            )
    elif dataset_name == "imagefolder64":
        if augment is None:
            augment = split == "train"
        transform = build_image_transform(
            image_size=image_size,
            normalize=normalize,
            augment=augment,
        )
        split_root = os.path.join(data_root, split)
        image_root = split_root if os.path.isdir(split_root) else data_root
        dataset = datasets.ImageFolder(image_root, transform=transform)
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Choose one of {SUPPORTED_DATASETS}."
        )

    if not 0 < subset_fraction <= 1:
        raise ValueError(
            "subset_fraction must be greater than 0 and less than or equal to 1."
        )
    if subset_fraction < 1:
        subset_size = max(1, int(len(dataset) * subset_fraction))
        generator = torch.Generator().manual_seed(subset_seed)
        indices = torch.randperm(
            len(dataset),
            generator=generator,
        )[:subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
