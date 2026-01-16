import torch
from torch import as_tensor
from torchvision.io import decode_jpeg

from .transforms import TransformFactoryWithSavedRandomValue


class VideoDecoderWithMask():
    def __init__(self, clip_sampler, clip_sampler_args, transform: TransformFactoryWithSavedRandomValue,) -> None:
        self.clip_sampler = clip_sampler
        self.clip_sampler_args = clip_sampler_args
        self.transform = transform
        self.frame_indices = None

    def update_frame_indices(self, frame_sec_list):
        self.frame_indices = self.clip_sampler(frame_sec_list, self.clip_sampler_args)

    def video_decoder(
        self,
        video_pickle,
        with_frame_sec=True,
        single_channel=False,
        update_transform_random_value=False,
        is_train=True,
        update_frame_indices=False,
    ):
        if with_frame_sec:
            jpg_byte_list, frame_sec_list = video_pickle
        else:
            jpg_byte_list = video_pickle
            frame_sec_list = [0 for _ in range(len(jpg_byte_list))]

        clip = [decode_jpeg(as_tensor(list(jpg_byte), dtype=torch.uint8)) for jpg_byte in jpg_byte_list]
        if single_channel:
            # マスク画像をデコード後，チャネル数が3になることがあるので、1チャネルにする
            clip = [frame[0].unsqueeze(0) for frame in clip]
        clip = torch.stack(clip, 0)  # TCHW
        if update_frame_indices:
            self.update_frame_indices(frame_sec_list)
        clip = clip[self.frame_indices]
        clip = torch.permute(clip, (1, 0, 2, 3))  # CTHW

        if self.transform is not None:
            if update_transform_random_value:
                self.transform.recreate_compose(
                    is_train=is_train,
                    input_img_height=clip.shape[2],
                    input_img_width=clip.shape[3])
            if single_channel:
                clip = self.transform.compose_for_mask(clip)
            else:
                clip = self.transform.compose_for_3ch(clip)

        return clip
