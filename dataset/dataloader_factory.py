from typing import Literal, Union, get_args
from dataclasses import dataclass
import argparse

from torch.utils.data import DataLoader

from dataset.wds_dataloader import return_dataloader

from dataset import (
    cifar10,
    Cifar10Info,
    cifar100,
    Cifar100Info,
    image_folder,
    ImageFolderInfo,
    video_folder,
    VideoFolderInfo,
    zero_images,
    ZeroImageInfo,
    transform_image,
    TransformImageInfo,
    transform_video,
    TransformVideoInfo,
    transform_rgb_edge,
    TransformRGBEdgeInfo,
    transform_for_hat,
    TransformHATInfo,
)

from dataset.wds_folder import wds_video_folder


@dataclass
class DataloadersInfo:
    """DataloadersInfo

        train_loader (torch.utils.data.DataLoader): training set loader
        val_loader (torch.utils.data.DataLoader): validation set loader
        n_classes (int): number of classes
    """
    train_loader: DataLoader
    val_loader: DataLoader
    n_classes: int


WebDatasets = Literal["Mimetics_wds", "UCF101_wds", "HMDB51_wds", "Kinetics400_wds", "UCF_hat", "mim_hat"]

SupportedDatasets = Union[Literal["CIFAR10", "CIFAR100", "ImageFolder",
                                  "VideoFolder", "ZeroImages"], WebDatasets]


def configure_dataloader(
    command_line_args: argparse.Namespace,
    dataset_name: SupportedDatasets,
):
    """dataloader factory

    Args:
        command_line_args (argparse.Namespace): command line args
        dataset_name (SupportedDatasets): dataset name (str).
            ["CIFAR10", "ImageFolder", "VideoFolder", "ZeroImages"]

    Raises:
        ValueError: invalid dataset_name is given

    Returns:
        (DataloadersInfo): dataset information
    """

    args = command_line_args

    if dataset_name == "CIFAR10":
        train_transform, val_transform = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            cifar10(Cifar10Info(
                root=args.root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    if dataset_name == "CIFAR100":
        train_transform, val_transform = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            cifar100(Cifar100Info(
                root=args.root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataset_name == "ImageFolder":
        train_transform, val_transform = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            image_folder(ImageFolderInfo(
                root=args.root,  # /mnt/NAS-TVS872XT/dataset/Tiny-ImageNet/tiny-imagenet-200
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataset_name == "VideoFolder":
        train_transform, val_transform = \
            transform_for_hat(TransformHATInfo(
                frames_per_clip=args.frames_per_clip
            ))
        train_loader, val_loader, n_classes = \
            video_folder(VideoFolderInfo(
                root=args.root,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform,
                clip_duration=args.clip_duration,
                clips_per_video=args.clips_per_video,
            ))

    elif dataset_name == "ZeroImages":
        train_transform, _ = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            zero_images(ZeroImageInfo(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                transform=train_transform,
            ))

    elif dataset_name == "UCF_hat" or dataset_name == "mim_hat":
        train_loader, val_loader, n_classes = \
            wds_video_folder(
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                clip_frames=args.frames_per_clip,
                clip_duration=args.clip_duration,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                gpus=args.devices
            )

    elif dataset_name in get_args(WebDatasets):
        train_loader, val_loader, n_classes = \
            return_dataloader(
                args,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
            )

    else:
        raise ValueError("invalid dataset_name")

    return DataloadersInfo(
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=n_classes
    )


def cut_strings_in_dict(class_to_ids, substring):
    new_class_to_ids = {}
    for key, value in class_to_ids.items():
        cut_index = key.find(substring)
        if cut_index != -1:
            new_key = key[:cut_index]
        else:
            new_key = key
        new_class_to_ids[new_key] = value
    return new_class_to_ids


def make_class_to_ids(loader, cls_id_to_cls_name_file=None):
    class_to_ids = loader.dataset.class_to_idx
    if cls_id_to_cls_name_file:
        # class_name_file = '/mnt/NAS-TVS872XT/dataset/Tiny-ImageNet/tiny-imagenet-200/words.txt'
        folder_class_dict = {}
        with open(cls_id_to_cls_name_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                folder_name = parts[0]
                class_name = parts[1]
                folder_class_dict[folder_name] = class_name
        folder_dict = {index: name for index, name in enumerate(class_to_ids.keys())}

        class_to_ids = {folder_class_dict[v]: k for k, v in folder_dict.items()}
        substring = ','
        class_to_ids = cut_strings_in_dict(class_to_ids, substring)
    return class_to_ids
