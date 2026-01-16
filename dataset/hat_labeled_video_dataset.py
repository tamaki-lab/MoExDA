# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations
from typing import List, Tuple, Optional, Callable, Any, Type
import torch
from pytorchvideo.transforms.functional import uniform_temporal_subsample

import gc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler

from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.utils import MultiProcessSampler

from pytorchvideo.data import LabeledVideoDataset


logger = logging.getLogger(__name__)


def _ensure_fixed_frames(video_tensor, frames_per_clip=16):
    """
    Ensure the video tensor has exactly `frames_per_clip` frames.
    If the video has fewer frames, pad with the last frame.
    If the video has more frames, truncate it.

    Args:
        video_tensor (torch.Tensor): The video tensor with shape (T, C, H, W).
        frames_per_clip (int): Number of frames to ensure.

    Returns:
        torch.Tensor: The processed video tensor with shape (frames_per_clip, C, H, W).
    """
    num_frames, c, h, w = video_tensor.size()

    if num_frames < frames_per_clip:
        # Pad with the last frame
        padding_frames = frames_per_clip - num_frames
        last_frame = video_tensor[-1:].expand(padding_frames, c, h, w)
        video_tensor = torch.cat((video_tensor, last_frame), dim=0)
    elif num_frames > frames_per_clip:
        # Truncate extra frames
        video_tensor = video_tensor[:frames_per_clip]

    return video_tensor


class HATLabeledVideoDataset(LabeledVideoDataset):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decode_video: bool = True,
        decoder: str = "pyav",
        bg_only_labeled_video_paths: Optional[List[Tuple[str, Optional[dict]]]] = None,
        human_seg_labeled_video_paths: Optional[List[Tuple[str, Optional[dict]]]] = None,
        frame_per_clip: int = 16,
    ) -> None:
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List of original video paths
                and associated labels.

            clip_sampler (ClipSampler): Defines how clips should be sampled from each video.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal video container.

            transform (Callable): Transform function for preprocessing or augmentations.

            decode_audio (bool): Whether to decode audio from videos.

            decode_video (bool): Whether to decode video frames from videos.

            decoder (str): Type of decoder to use for decoding video.

            bg_only_labeled_video_paths (List[Tuple[str, Optional[dict]]], optional):
                List of 'background-only' video paths and associated labels.

            human_seg_labeled_video_paths (List[Tuple[str, Optional[dict]]], optional):
                List of 'human-segmentation' video paths and associated labels.
        """
        # 親クラスの初期化を呼び出し
        super().__init__(
            labeled_video_paths=labeled_video_paths,
            clip_sampler=clip_sampler,
            video_sampler=video_sampler,
            transform=transform,
            decode_audio=decode_audio,
            decode_video=decode_video,
            decoder=decoder,
        )

        # 子クラス特有の初期化
        self.bg_only_labeled_video_paths = bg_only_labeled_video_paths
        self.human_seg_labeled_video_paths = human_seg_labeled_video_paths

        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, decode audio from video.

            decode_video (bool): If True, decode video frames from a video container.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._decode_video = decode_video
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._bg_only_labeled_videos = bg_only_labeled_video_paths
        self._human_seg_labeled_videos = human_seg_labeled_video_paths
        self._decoder = decoder
        self._frames_per_clip = frame_per_clip

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        # Initialize samplers for bg_only and human_seg datasets
        self._bg_video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._bg_video_random_generator = torch.Generator()
            self._bg_video_sampler = video_sampler(
                self.bg_only_labeled_video_paths, generator=self._bg_video_random_generator
            )
        else:
            self._bg_video_sampler = video_sampler(self.bg_only_labeled_video_paths)

        self._human_seg_video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._human_seg_video_random_generator = torch.Generator()
            self._human_seg_video_sampler = video_sampler(
                self.human_seg_labeled_video_paths, generator=self._human_seg_video_random_generator
            )
        else:
            self._human_seg_video_sampler = video_sampler(self.human_seg_labeled_video_paths)

        # Initialize video sampler iterators
        self._video_sampler_iter = None  # Initialized on first call to self.__next__()
        self._bg_video_sampler_iter = None
        self._human_seg_video_sampler_iter = None

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._bg_loaded_video_label = None
        self._human_seg_loaded_video_label = None

        self._loaded_clip = None
        self._bg_loaded_clip = None
        self._human_seg_loaded_clip = None
        self._last_clip_end_time = None
        self._bg_last_clip_end_time = None
        self.video_path_handler = VideoPathHandler()

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        if not self._bg_video_sampler_iter:
            # Setup MultiProcessSampler for the bg_only video sampler
            self._bg_video_sampler_iter = iter(MultiProcessSampler(self._bg_video_sampler))

        # Note: human_seg_video uses the same sampler as video to ensure they match.
        if not self._human_seg_video_sampler_iter:
            self._human_seg_video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Get the next video and its label
            video_index = next(self._video_sampler_iter)
            video_path, info_dict = self._labeled_videos[video_index]
            video_label = info_dict["label"]

            try:
                # human_seg_video: same as video
                human_seg_video_index = video_index  # Ensure the same video is selected
                human_seg_video_path, human_seg_info_dict = self._human_seg_labeled_videos[human_seg_video_index]
            except IndexError:
                raise RuntimeError(f"No human_seg_video found for index {video_index}")

            try:
                # bg_only_video: randomly sampled
                bg_video_index = next(self._bg_video_sampler_iter)
                bg_video_path, bg_info_dict = self._bg_only_labeled_videos[bg_video_index]
            except StopIteration:
                raise RuntimeError("No bg_only_video available to sample.")

            # Load the videos
            video = self.video_path_handler.video_from_path(
                video_path,
                decode_audio=self._decode_audio,
                decode_video=self._decode_video,
                decoder=self._decoder,
            )
            human_seg_video = self.video_path_handler.video_from_path(
                human_seg_video_path,
                decode_audio=self._decode_audio,
                decode_video=self._decode_video,
                decoder=self._decoder,
            )
            bg_video = self.video_path_handler.video_from_path(
                bg_video_path,
                decode_audio=self._decode_audio,
                decode_video=self._decode_video,
                decoder=self._decoder,
            )

            # Perform clip sampling
            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(self._last_clip_end_time, video.duration, info_dict)
            (
                bg_clip_start,
                bg_clip_end,
                bg_clip_index,
                bg_aug_index,
                bg_is_last_clip,
            ) = self._clip_sampler(self._last_clip_end_time, bg_video.duration, bg_info_dict)

            # Get clips for human_seg_video and bg_only_video
            human_seg_clip = human_seg_video.get_clip(clip_start, clip_end)
            bg_clip = bg_video.get_clip(bg_clip_start, bg_clip_end)
            frames = video.get_clip(clip_start, clip_end)
            bg_info_dict = {f"bg_{key}": value for key, value in bg_info_dict.items()}
            human_seg_info_dict = {f"human_seg_{key}": value for key, value in human_seg_info_dict.items()}

            raw_frames = frames["video"]
            uniformed_video = uniform_temporal_subsample(raw_frames, self._frames_per_clip)
            raw_bg_frames = bg_clip["video"]
            uniformed_bg_video = uniform_temporal_subsample(raw_bg_frames, self._frames_per_clip)
            raw_human_seg_frames = human_seg_clip["video"]
            uniformed_human_seg_video = uniform_temporal_subsample(raw_human_seg_frames, self._frames_per_clip)

            # Prepare the output dictionary
            sample_dict = {
                "video": uniformed_video.permute(1, 0, 2, 3),
                "bg_video": uniformed_bg_video.permute(1, 0, 2, 3),
                "human_seg_video": uniformed_human_seg_video.permute(1, 0, 2, 3),
                "video_name": video.name,
                "bg_video_name": bg_video.name,
                "human_seg_video_name": human_seg_video.name,
                "video_index": video_index,
                "bg_video_index": bg_video_index,
                "human_seg_video_index": human_seg_video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                **bg_info_dict,
                **human_seg_info_dict,
            }

            if self._transform is not None:
                # sample_dict["video"] = self._transform(sample_dict["video"])
                # sample_dict["bg_video"] = self._transform(sample_dict["bg_video"])
                # sample_dict["human_seg_video"] = self._transform(sample_dict["human_seg_video"])
                sample_dict = self._transform(sample_dict)

            return sample_dict

        raise RuntimeError(
            f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
        )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


ORIGINAL = "original_videos"
BG_ONLY = "person_inpainting_videos"
HUMAN_SEG = "person_segment_mask_videos"


def hat_labeled_video_data(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> HATLabeledVideoDataset:
    """
    A helper function to create `LabeledVideoDataset object for Ucf101 and Kinetics datasets.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the `LabeledVideoDataset class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in `LabeledVideoDataset. All the video paths before loading
                are prefixed with this path.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

    """
    original_path = data_path + '/' + ORIGINAL
    labeled_video_paths = LabeledVideoPaths.from_path(original_path)
    labeled_video_paths.path_prefix = video_path_prefix

    bg_only_path = data_path + '/' + BG_ONLY
    bg_only_labeled_video_paths = LabeledVideoPaths.from_path(bg_only_path)
    bg_only_labeled_video_paths.path_prefix = video_path_prefix

    human_seg_path = data_path + '/' + HUMAN_SEG
    human_seg_labeled_video_paths = LabeledVideoPaths.from_path(human_seg_path)
    human_seg_labeled_video_paths.path_prefix = video_path_prefix

    dataset = HATLabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
        bg_only_labeled_video_paths=bg_only_labeled_video_paths,
        human_seg_labeled_video_paths=human_seg_labeled_video_paths,
        frame_per_clip=16
    )
    return dataset
