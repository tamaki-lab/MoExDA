from dataclasses import dataclass
from typing import Tuple
import av
import numpy as np
import torch
import torchvision
import kornia
import random
import torchvision.transforms.v2 as image_transform
import torchvision.transforms.functional as F
import torchvision.io as io
import matplotlib.pyplot as plt


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


@dataclass
class TransformRGBEdgeInfo:
    frames_per_clip: int = 16
    resize_size: int = 256
    crop_size: int = 224
    val_shorter_side_size: int = 256
    min_shorter_side_size: int = 256
    max_shorter_side_size: int = 320


def mp4_to_tensor(file_path):
    container = av.open(file_path)
    frames = []

    for frame in container.decode(video=0):
        frame = frame.to_rgb().to_ndarray()
        frame = frame.astype(np.uint8)
        frames.append(frame)

    video_tensor = torch.tensor(np.array(frames))
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (フレーム数, チャンネル数, 高さ, 幅)
    return video_tensor


def calc_hist(edge_tensor, rgb_tensor):
    T, C, H, W = edge_tensor.shape
    T, C, H, W = rgb_tensor.shape
    assert C == 3, "RGBフレームは3チャネルである必要があります!"

    frame_indices = range(T)
    histograms = {
        "r": np.zeros(256),
        "g": np.zeros(256),
        "b": np.zeros(256),
        "edges": np.zeros(256),
    }

    for idx in frame_indices:
        frame = rgb_tensor[idx].permute(1, 2, 0).cpu().numpy()
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)

        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        histograms["b"] += np.histogram(b, bins=256, range=(0, 256))[0]
        histograms["g"] += np.histogram(g, bins=256, range=(0, 256))[0]
        histograms["r"] += np.histogram(r, bins=256, range=(0, 256))[0]

        edge_frame = edge_tensor[idx, 0].cpu().numpy()
        if edge_frame.max() <= 1.0:
            edge_frame = (edge_frame * 255).astype(np.uint8)
        histograms["edges"] += np.histogram(edge_frame, bins=256, range=(0, 256))[0]

        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.plot(histograms["r"], color='r', label="r")
        ax1.plot(histograms["g"], color='g', label="g")
        ax1.plot(histograms["b"], color='b', label="b")
        ax1.legend()
        ax1.set_title("RGB Histogram")
        ax1.set_yticks([])  # 縦軸の目盛りをオフ
        ax1.set_xlim(0, 255)
        ax1.set_ylim(0, max(histograms["r"].max(), histograms["g"].max(), histograms["b"].max()) * 1.1)
        ax1.set_aspect('auto')
        plt.tight_layout()
        plt.savefig("rgb_hist.jpg")

        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.plot(histograms["edges"], color="k", label="edges")
        for i, value in enumerate(histograms["edges"]):
            if value > 0:  # ヒストグラムが 0 より大きい場合に垂直線を描画
                ax2.vlines(x=i, ymin=0, ymax=value, color='k', alpha=0.5)
        ax2.legend()
        ax2.set_title("Edge Histogram")
        ax2.set_yticks([])  # 縦軸の目盛りをオフ
        ax2.set_xlim(0, 255)
        ax2.set_ylim(0, histograms["edges"].max() * 1.1)
        ax2.set_aspect('auto')  # 正方形に依存しない表示

        plt.tight_layout()
        plt.savefig("edge_hist.jpg")


if __name__ == '__main__':
    file_path = "/mnt/HDD10TB-1/sugimoto/2024_sugimoto_edge/datasets/HMDB51/test/stand/CastAway1_stand_f_nm_np1_le_med_24.mp4"
    video_tensor = mp4_to_tensor(file_path)
    identity_frame = Identity()
    sobel_filter = kornia.filters.Sobel()
    grayscale = kornia.color.RgbToGrayscale()
    transform = image_transform.Compose([
        image_transform.ToImage(),  # HWC ndarray --> CHW tensor (Image)
        image_transform.ToDtype(torch.float32, scale=True),  # [0,255] --> [0,1]
        DataSplit(),
        MapSplitData(
            grayscale,
            identity_frame,
        ),
        MapSplitData(
            sobel_filter,
            identity_frame,
        )
    ])
    edge_tensor, rgb_tensor = transform(video_tensor)
    edge_tensor = edge_tensor.repeat(1, 3, 1, 1)
    # io.write_video("edge.mp4", edge_tensor.permute(0, 2, 3, 1) * 255, fps=25, video_codec="h264")
    # io.write_video("rgb.mp4", rgb_tensor.permute(0, 2, 3, 1) * 255, fps=25, video_codec="h264")
    calc_hist(edge_tensor, rgb_tensor)
    # print(edge_tensor.shape)
    # print(type(edge_tensor))
