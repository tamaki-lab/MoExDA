from dataclasses import dataclass
from typing import Tuple

import torch
import torchvision.transforms.v2 as image_transform
import pytorchvideo.transforms as video_transform
from dataset.transforms import TransformRGBEdgeInfo
import kornia
import torchvision.transforms.functional as F
import torchvision
import random


def transform_video(
        trans_info: TransformRGBEdgeInfo
) -> Tuple[image_transform.Compose, image_transform.Compose]:
    """transform for images

    Args:
        trans_image_info (TransformImageInfo): information for image transform

    Returns:
        Tuple[torchvision.transforms]: train and val transforms
    """

    # train_transform = image_transform.Compose([
    #     image_transform.ToImage(),  # HWC ndarray --> CHW tensor (Image)
    #     image_transform.ToDtype(torch.float32, scale=True),  # [0,255] --> [0,1]
    #     image_transform.RandomResizedCrop(trans_info.crop_size, antialias=True),
    #     image_transform.RandomHorizontalFlip(),
    #     image_transform.Normalize(
    #         [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    #     ),
    # ])

    # val_transform = image_transform.Compose([
    #     image_transform.ToImage(),
    #     image_transform.ToDtype(torch.float32, scale=True),
    #     image_transform.Resize(trans_info.resize_size, antialias=True),
    #     image_transform.CenterCrop(trans_info.crop_size),
    #     image_transform.Normalize(
    #         [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    #     ),
    # ])
    # return train_transform, val_transform

    sobel_filter = kornia.filters.Sobel()
    grayscale = kornia.color.RgbToGrayscale()
    rgb_norm = image_transform.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    sobel_norm = image_transform.Normalize(
        [0.026, 0.026, 0.026], [0.037, 0.037, 0.037]
    )
    # rgb_norm = image_transform.Normalize(
    #     [0, 0, 0], [1, 1, 1]
    # )
    # sobel_norm = image_transform.Normalize(
    #     [0, 0, 0], [1, 1, 1]
    # )
    identity_frame = Identity()
    center_crop = image_transform.CenterCrop(trans_info.crop_size)

    train_transform = image_transform.Compose([
        image_transform.ToImage(),  # HWC ndarray --> CHW tensor (Image)
        image_transform.ToDtype(torch.float32, scale=True),  # [0,255] --> [0,1]
        video_transform.RandomShortSideScale(
            min_size=trans_info.min_shorter_side_size,
            max_size=trans_info.max_shorter_side_size,
        ),
        DataSplit(),
        MapSplitData(
            grayscale,
            identity_frame,
        ),
        MapSplitData(
            sobel_filter,
            identity_frame,
        ),
        SameRandomCrop(
            trans_info.crop_size
        ),
        SameRandomHorizontalFlip(),
        MapSplitData(
            sobel_norm,
            rgb_norm,
        ),
    ])

    val_transform = image_transform.Compose([
        image_transform.ToImage(),
        image_transform.ToDtype(torch.float32, scale=True),
        video_transform.ShortSideScale(
            trans_info.val_shorter_side_size,
        ),
        DataSplit(),
        MapSplitData(
            grayscale,
            identity_frame,
        ),
        MapSplitData(
            sobel_filter,
            identity_frame,
        ),
        MapSplitData(
            center_crop,
            center_crop,
        ),
        MapSplitData(
            sobel_norm,
            rgb_norm,
        ),
    ])

    return train_transform, val_transform


class Identity():
    def __call__(self, x):
        return x


class DataSplit():
    def __call__(self, x):
        return x, x


class SameRandomCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # クロップのランダムな位置(i, j)
        # クロップサイズ(h,w)
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(x[0], output_size=(self.size, self.size))
        return F.crop(x[0], i, j, h, w), F.crop(x[1], i, j, h, w)


class SameRandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p  # フリップの確率

    def __call__(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # ランダムにフリップするかどうかを決定
        flip = random.random() < self.p  # random.random()は[0,1)の範囲でランダムな実数を返す

        if flip:
            return F.hflip(x[0]), F.hflip(x[1])
        else:
            return x[0], x[1]


class MapSplitData():
    def __init__(self, func1, func2):
        self.func1 = func1
        self.func2 = func2

    def __call__(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.func1(x[0]), self.func2(x[1])
