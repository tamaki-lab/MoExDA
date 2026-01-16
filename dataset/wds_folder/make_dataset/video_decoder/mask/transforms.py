import math
import torch
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    ToDtype,
    functional as tv_f,
)
from pytorchvideo.transforms import (
    Normalize,
    ShortSideScale,
)
from torchvision import ops


class Crop(torch.nn.Module):
    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):
        return tv_f.crop(img, self.top, self.left, self.height, self.width)


class TransformFactoryWithSavedRandomValue:
    """
    複数の動画に対して同様の変換を行うためのクラス（ex. 元動画, マスク動画に同様の処理をする）
    ランダムなパラメータを保持. 更新するときはreset_random_sizeを呼ぶ
    """

    def __init__(self):
        self.scale_min = 226
        self.scale_max = 256
        self.crop_size = 224
        self.compose_for_3ch = None
        self.compose_for_mask = None

    def recreate_compose(self, is_train=True, input_img_height=None, input_img_width=None):
        if is_train:
            assert (
                input_img_height is not None and input_img_width is not None
            ), "input_img_height and input_img_width must be specified"
            ih, iw = input_img_height, input_img_width
            short_side_size = torch.randint(self.scale_min, self.scale_max + 1, (1,)).item()

            if ih < iw:
                scaled_height = short_side_size
                scaled_width = int(math.floor((float(iw) / ih) * short_side_size))
            else:
                scaled_height = int(math.floor((float(ih) / iw) * short_side_size))
                scaled_width = short_side_size

            top, left = self._get_random_crop_params(scaled_height, scaled_width)
            self.compose_for_3ch, self.compose_for_mask = self._create(
                is_train=True,
                scale_size=short_side_size,
                crop_size=self.crop_size,
                crop_top=top,
                crop_left=left,
            )
        else:  # validation
            self.compose_for_3ch, self.compose_for_mask = self._create(
                is_train=False, scale_size=self.scale_min, crop_size=self.crop_size
            )

    def _get_random_crop_params(self, input_img_height, input_img_width) -> (int, int):
        if input_img_height < self.crop_size or input_img_width < self.crop_size:
            raise ValueError(f"crop_size {(self.crop_size, self.crop_size)} must be smaller than input_image_size {
                (input_img_height, input_img_width)}")

        if input_img_width == self.crop_size and input_img_height == self.crop_size:
            return 0, 0, input_img_height, input_img_width

        y = torch.randint(0, input_img_height - self.crop_size + 1, size=(1,)).item()
        x = torch.randint(0, input_img_width - self.crop_size + 1, size=(1,)).item()
        return y, x

    def _create(
        self,
        is_train=True,
        scale_size=None,
        crop_size=None,
        crop_top=None,
        crop_left=None,
    ):
        transform_list = [
            ToDtype(torch.float32, scale=True),
            Normalize(
                [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
            ),
        ]
        transform_list_for_mask = [
            ToDtype(torch.float32, scale=True),
        ]

        if is_train:
            tl = [
                ShortSideScale(scale_size),
                Crop(crop_top, crop_left, crop_size, crop_size),
                ops.Permute([1, 0, 2, 3]),  # CTHW --> TCHW
            ]
        else:
            tl = [
                ShortSideScale(scale_size),
                CenterCrop(crop_size),
                ops.Permute([1, 0, 2, 3]),
            ]
        transform_list.extend(tl)
        transform_list_for_mask.extend(tl)

        transform = Compose(transform_list)
        transform_for_mask = Compose(transform_list_for_mask)
        return transform, transform_for_mask
