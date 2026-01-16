from pathlib import Path
import webdataset as wds
from functools import partial

from .make_dataset.wds_dataset import make_dataset
from .info_from_json import info_from_json
from .clip_sampler import uniform_clip_sampler, center_frame_clip_sampler, random_start_frame_clip_sampler

from .make_dataset.wds_dataset import ShardsType

import torch


def get_num_gpus(devices):
    if devices == "-1":
        num_gpus = torch.cuda.device_count()
    elif devices.isdigit():
        num_gpus = int(devices)
    else:
        gpu_ids = devices.split(",")
        num_gpus = len(gpu_ids)

    return num_gpus


def wds_video_folder(
    train_dir,
    val_dir,
    clip_frames,
    clip_duration,
    batch_size,
    num_workers,
    gpus,
):
    make_loader = partial(
        wds_video_dataloader,
        clip_frames=clip_frames,
        clip_duration=clip_duration,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_paths = val_dir.split(",")
    train_loader = make_loader(train_dir, is_train=True, gpus=gpus)
    val_loaders = [
        make_loader(p, is_train=False, gpus=gpus) for p in val_paths
    ]
    _, n_classes, _ = info_from_json(train_dir)

    return train_loader, val_loaders, n_classes


def wds_video_dataloader(
    shards_path,
    is_train,
    clip_frames,
    clip_duration,
    batch_size,
    num_workers,
    gpus,
):
    shards_type: ShardsType = ShardsType.STANDARD
    if ShardsType.ACTIONSWAP.value in shards_path:
        shards_type = ShardsType.ACTIONSWAP
    elif ShardsType.MASK.value in shards_path:
        shards_type = ShardsType.MASK

    shards_path_list = [str(path) for path in Path(shards_path).glob("*.tar") if not path.is_dir()]
    dataset_size, _, _ = info_from_json(shards_path)

    clip_sampler = random_start_frame_clip_sampler

    # if clip_sampler == "uniform":
    #     clip_sampler = uniform_clip_sampler
    # elif clip_sampler == "center_frame":
    #     clip_sampler = center_frame_clip_sampler
    # elif clip_sampler == "random":
    #     clip_sampler = random_start_frame_clip_sampler

    num_gpus = get_num_gpus(gpus)

    dataset, collate_fn = make_dataset(
        is_train=is_train,
        shards_url=shards_path_list,
        clip_sampler=clip_sampler,
        clip_sampler_args={
            "clip_frames": clip_frames,
            "clip_duration": clip_duration,
        },
        shuffle_buffer_size=100,
        is_ddp=num_gpus > 1,
        shards_type=shards_type,
    )
    dataset = dataset.batched(batch_size, partial=False)

    loader = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        shuffle=False,
        batch_size=None,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    num_batches = dataset_size // (batch_size * num_gpus)
    loader.length = num_batches
    loader = loader.with_length(num_batches)
    loader = loader.repeat(nbatches=num_batches)
    loader = loader.slice(num_batches)  # pylint: disable=E1101
    # loader = loader.ddp_equalize(dataset_size // self.args.batch_size)

    return loader
