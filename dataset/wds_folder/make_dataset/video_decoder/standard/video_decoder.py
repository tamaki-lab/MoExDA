import torch
from torch import as_tensor
from torchvision.io import decode_jpeg, decode_png


class VideoDecoder():
    def __init__(self, clip_sampler, clip_sampler_args, transform) -> None:
        self.clip_sampler = clip_sampler
        self.clip_sampler_args = clip_sampler_args
        self.transform = transform
        self.frame_indices = None

    def video_decoder(
        self,
        video_pickle,
    ):
        if len(video_pickle) == 2:
            jpg_byte_list, frame_sec_list = video_pickle
        else:
            jpg_byte_list = video_pickle
            frame_sec_list = [0 for _ in range(len(jpg_byte_list))]

        frame_indices = self.clip_sampler(frame_sec_list, self.clip_sampler_args)

        clip = [decode_jpeg(as_tensor(list(jpg_byte_list[i]),
                                      dtype=torch.uint8))
                for i in frame_indices]
        clip = torch.stack(clip, 0)  # TCHW

        if self.transform is not None:
            clip = self.transform(clip)

        return clip
