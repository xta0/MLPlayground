import torch
import torchvision
import torchvision.transforms as transforms

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def build_transform(train=False):
    """Build shared CIFAR-10 preprocessing, with augmentation for training."""
    steps = [transforms.ToTensor()]
    if train:
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                (32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333),
                interpolation=transforms.InterpolationMode.BILINEAR),
        ])
    steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(steps)


def create_data_loader(batch_size=32, num_workers=2, *, train=False,
                       root="./data", download=True):
    """Load one CIFAR-10 split; only the training split is augmented and shuffled."""
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train, download=download, transform=build_transform(train))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


def prepare_data(batch_size=32, num_workers=2, *, root="./data", download=True):
    """Return training loader, test loader, and class names."""
    trainloader = create_data_loader(
        batch_size, num_workers, train=True, root=root, download=download)
    testloader = create_data_loader(
        batch_size, num_workers, train=False, root=root, download=download)
    return trainloader, testloader, CLASSES
