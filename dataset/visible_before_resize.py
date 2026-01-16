import os
import av
import torch
import numpy as np
import torchvision.io as io
import kornia
import torchvision.transforms.v2 as image_transform
import pytorchvideo.transforms as video_transform
import torchvision.transforms.functional as TF


class ResizeToEven:
    def __call__(self, frame):
        h, w = frame.shape[-2], frame.shape[-1]
        new_h = h if h % 2 == 0 else h + 1
        new_w = w if w % 2 == 0 else w + 1
        return TF.resize(frame, (new_h, new_w))


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


def process_videos_in_directory(directory_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    video_files = [f for f in os.listdir(directory_path) if f.endswith('.mp4')]
    video_files = video_files[:50]

    transform = image_transform.Compose([
        image_transform.ToImage(),  # HWC ndarray --> CHW tensor (Image)
        image_transform.ToDtype(torch.float32, scale=True),  # [0,255] --> [0,1]
        kornia.color.RgbToGrayscale(),
        kornia.filters.Sobel(),
        video_transform.RandomShortSideScale(
            min_size=256,
            max_size=320,
        ),
        # image_transform.RandomCrop((224, 224)),
        ResizeToEven(),
        image_transform.Normalize([0, 0, 0], [1, 1, 1])
    ])

    for video_file in video_files:
        input_path = os.path.join(directory_path, video_file)
        output_path = os.path.join(output_directory, f"processed_{video_file}")
        video_tensor = mp4_to_tensor(input_path)
        video_tensor = transform(video_tensor)
        video_tensor = video_tensor.permute(0, 2, 3, 1) * 255.0

        io.write_video(output_path, video_tensor, fps=30, video_codec="h264")
        print(f"Processed and saved: {output_path}")


# 処理するディレクトリのパス
input_directory = "/mnt/NAS-TVS872XT/dataset/Kinetics400/train/abseiling"
output_directory = "../visible_before_resize"

# ディレクトリ内の動画を処理
process_videos_in_directory(input_directory, output_directory)
