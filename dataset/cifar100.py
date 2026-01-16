from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.datasets import CIFAR100


@dataclass
class Cifar100Info():
    root: str
    batch_size: int
    num_workers: int
    train_transform: transforms
    val_transform: transforms


def cifar100(
        cifar100_info: Cifar100Info
):

    train_dataset = CIFAR100(
        root=cifar100_info.root,
        train=True,
        download=True,
        transform=cifar100_info.train_transform)
    val_dataset = CIFAR100(
        root=cifar100_info.root,
        train=False,
        download=True,
        transform=cifar100_info.val_transform)
    n_classes = 100

    train_loader = DataLoader(
        train_dataset,
        batch_size=cifar100_info.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cifar100_info.num_workers)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cifar100_info.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cifar100_info.num_workers)

    return train_loader, val_loader, n_classes
