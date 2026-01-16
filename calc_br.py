import av
import numpy as np
import torch
from pathlib import Path


def mp4_to_tensor(file_path):
    container = av.open(file_path)
    frames = []

    for frame in container.decode(video=0):
        frame = frame.to_rgb().to_ndarray()
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    video_tensor = torch.tensor(np.array(frames))
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (フレーム数, チャンネル数, 高さ, 幅)
    return video_tensor


def calc_rgb_mean(file):
    videos = list(Path(file).iterdir())
    r_list = []
    g_list = []
    b_list = []
    for video in videos:
        video = str(video)
        video_tensor = mp4_to_tensor(video)
        # print(video_tensor.shape) TCHW
        mean_tensor = video_tensor.mean(dim=(2, 3))  # HWに関して平均を取る
        mean_tensor = mean_tensor.mean(dim=0)  # フレーム方向に平均を取る

        r_tensor = mean_tensor[0]
        g_tensor = mean_tensor[1]
        b_tensor = mean_tensor[2]

        r = float(r_tensor)
        g = float(g_tensor)
        b = float(b_tensor)

        r_list.append(r)
        g_list.append(g)
        b_list.append(b)

    r_mean = sum(r_list) / len(r_list)
    g_mean = sum(g_list) / len(g_list)
    b_mean = sum(b_list) / len(b_list)

    return r_mean, g_mean, b_mean


def calc_rgb_std(file):
    videos = list(Path(file).iterdir())
    r_list = []
    g_list = []
    b_list = []
    for video in videos:
        video = str(video)
        video_tensor = mp4_to_tensor(video)
        std_tensor = video_tensor.std(dim=(2, 3), keepdim=False)  # HWに関して標準偏差を取る
        std_tensor = std_tensor.mean(dim=0)  # フレーム方向に関して平均を取る

        r_tensor = std_tensor[0]
        g_tensor = std_tensor[1]
        b_tensor = std_tensor[2]

        r = float(r_tensor)
        g = float(g_tensor)
        b = float(b_tensor)

        r_list.append(r)
        g_list.append(g)
        b_list.append(b)

    r_std = sum(r_list) / len(r_list)
    g_std = sum(g_list) / len(g_list)
    b_std = sum(b_list) / len(b_list)

    return r_std, g_std, b_std


if __name__ == '__main__':
    file_path = "/mnt/HDD10TB-1/sugimoto/2024_sugimoto_edge/visible_after_resize_original_Scaling"
    r_mean, g_mean, b_mean = calc_rgb_mean(file_path)
    r_std, g_std, b_std = calc_rgb_std(file_path)
    print(f"r_mean, g_mean, b_mean: {r_mean}, {g_mean}, {b_mean}")
    print(f"r_std, g_std, b_std: {r_std}, {g_std}, {b_std}")
