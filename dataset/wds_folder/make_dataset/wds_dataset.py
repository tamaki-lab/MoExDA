import random
from enum import Enum


import webdataset as wds

from .wds_pipeline import (
    StandardWdsPipeline,
    ActionSwapWdsPipeline,
    MaskWdsPipeline,
    WdsPipelineInterface,
)

from .video_decoder import (
    VideoDecoder,
    VideoDecoderWithMask,
    TransformFactoryWithSavedRandomValue,
    transform_video,
)

from dataset.transforms import TransformRGBEdgeInfo


class ShardsType(Enum):
    STANDARD = "data/wds/standard"
    ACTIONSWAP = "data/wds/actionswap"
    MASK = "data/wds/mask"


def make_dataset(
    is_train,
    shards_url,
    clip_sampler,
    clip_sampler_args,
    shuffle_buffer_size,
    # caption_text,
    is_ddp,
    shards_type: ShardsType,
):
    nodesplitter = wds.split_by_node if is_ddp else wds.single_node_only
    if is_train:
        random.shuffle(shards_url)
    # dataset = wds.WebDataset(shards_url, nodesplitter=nodesplitter, shardshuffle=subset)
    dataset = wds.WebDataset(shards_url, nodesplitter=nodesplitter)
    dataset = dataset.shuffle(shuffle_buffer_size) if is_train else dataset

    wds_pipeline: WdsPipelineInterface
    if shards_type == ShardsType.STANDARD or shards_type == ShardsType.ACTIONSWAP:
        if shards_type == ShardsType.STANDARD:
            wds_pipeline = StandardWdsPipeline()
        else:
            wds_pipeline = ActionSwapWdsPipeline()

        train_t, val_t = transform_video(TransformRGBEdgeInfo())
        transform = train_t if is_train else val_t
        video_decoder = VideoDecoder(
            clip_sampler=clip_sampler,
            clip_sampler_args=clip_sampler_args,
            transform=transform,
        )
    elif shards_type == ShardsType.MASK:
        wds_pipeline = MaskWdsPipeline()
        transform = TransformFactoryWithSavedRandomValue()
        video_decoder = VideoDecoderWithMask(
            clip_sampler=clip_sampler,
            clip_sampler_args=clip_sampler_args,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown shards_type: {shards_type}")

    dataset = wds_pipeline(dataset, video_decoder)
    collate_fn = wds_pipeline.collate_fn

    return dataset, collate_fn
