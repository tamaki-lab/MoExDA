
from functools import partial
import torch
import json
import webdataset as wds
from pathlib import Path
import pickle

from torchvision.io import decode_jpeg
from torch import as_tensor
from .clip_sampler import random_clip_sampler
from .transforms import transform_image, TransformImageInfo, transform_edge, TransformEdgeImageInfo, transform_rgb_edge, TransformRGBEdgeInfo
from args import ArgParse

args = ArgParse.get()


def info_from_json(shard_path):
    json_file = Path(shard_path).glob('*.json')
    json_file = str(next(json_file))  # get the first json file
    with open(json_file, 'r') as f:
        info_dic = json.load(f)

    dataset_size = info_dic['dataset size']

    if 'n_classes' in info_dic:
        n_classes = info_dic['n_classes']
        return dataset_size, n_classes
    else:
        return dataset_size


def video_decorder(video_pickle,
                   clip_sampler,
                   clip_sampler_args,
                   transform,
                   ):
    model_name = args.model_name
    jpg_byte_list, frame_sec_list = video_pickle

    frame_indices = clip_sampler(frame_sec_list, clip_sampler_args)

    clip = [decode_jpeg(as_tensor(list(jpg_byte_list[i]),
                                  dtype=torch.uint8))
            for i in frame_indices]
    clip = torch.stack(clip, 0)  # TCHW

    if transform is not None:
        clip = transform(clip)
        if model_name == 'x3d':
            clip = torch.permute(clip, (1, 0, 2, 3))  # CTHW

    return clip


def make_dataset(
    shards_url,
    dataset_size,
    clip_sampler,
    clip_sampler_args,
    transform,
    shuffle_buffer_size=100,
):
    if not args.use_moex:
        decode_video = partial(
            video_decorder,
            clip_sampler=clip_sampler,
            clip_sampler_args=clip_sampler_args,
            transform=transform,
        )

        dataset = wds.WebDataset(shards_url, nodesplitter=wds.split_by_node)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.decode(
            wds.handle_extension('stats.json', lambda x: json.loads(x)),
        )
        dataset = dataset.to_tuple(
            'video.pickle',
            'stats.json',
        )
        dataset = dataset.map_tuple(
            decode_video,  # 'video.bin'
            lambda x: x['label'],  # 1st 'stats.json'
        )
        dataset = dataset.with_length(dataset_size)
    elif args.use_moex:
        edge_rgb_transform = transform
        decode_video = partial(
            video_decorder,
            clip_sampler=clip_sampler,
            clip_sampler_args=clip_sampler_args,
            transform=edge_rgb_transform,
        )

        dataset = wds.WebDataset(shards_url, nodesplitter=wds.split_by_node)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.decode(
            wds.handle_extension('stats.json', lambda x: json.loads(x)),
        )
        dataset = dataset.to_tuple(
            'video.pickle',
            'stats.json',
            'stats.json',
        )
        dataset = dataset.map_tuple(
            decode_video,  # 'video.bin'
            lambda x: x['label'],  # 1st 'stats.json'
            lambda x: x['filename'],
        )
        dataset = dataset.with_length(dataset_size)

    return dataset


def return_dataloader(args, num_workers, batch_size, gpus=3):
    train_shards_path = [
        str(path) for path in Path(args.shards_path + '/train').glob('*.tar')
        if not path.is_dir()
    ]

    if not args.mimetics_path:
        val_shards_path = [
            str(path) for path in Path(args.shards_path + '/val').glob('*.tar')
            if not path.is_dir()
        ]
    else:
        val_shards_path = [
            str(path) for path in Path(args.mimetics_path).glob('*.tar')
            if not path.is_dir()
        ]

    if args.use_edge:
        train_transform, val_transform = transform_edge(TransformEdgeImageInfo())
    elif args.use_moex:
        train_transform, val_transform = transform_rgb_edge(TransformRGBEdgeInfo())
    else:
        train_transform, val_transform = transform_image(TransformImageInfo())

    train_dataset_size, n_classes = info_from_json(args.shards_path + '/train')

    if not args.mimetics_path:
        val_dataset_size = info_from_json(args.shards_path + '/val')[0]
    else:
        val_dataset_size = info_from_json(args.mimetics_path)[0]

    train_dataset = make_dataset(
        shards_url=train_shards_path,
        dataset_size=train_dataset_size,
        clip_sampler=random_clip_sampler,
        # clip_sampler=uniform_clip_sampler,
        clip_sampler_args={
            'clip_frames': args.frames_per_clip,
            'clip_duration': args.clip_duration,
        },
        transform=train_transform,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    val_dataset = make_dataset(
        shards_url=val_shards_path,
        dataset_size=val_dataset_size,
        clip_sampler=random_clip_sampler,
        # clip_sampler=uniform_clip_sampler,
        clip_sampler_args={
            'clip_frames': args.frames_per_clip,
            'clip_duration': args.clip_duration,
        },
        transform=val_transform,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    train_dataset = train_dataset.batched(
        batch_size,
        partial=False)

    val_dataset = val_dataset.batched(
        batch_size,
        partial=False)

    train_loader = wds.WebLoader(
        train_dataset,
        num_workers=num_workers,
        shuffle=False,
        batch_size=None,
        pin_memory=True,
    )

    val_loader = wds.WebLoader(
        val_dataset,
        num_workers=num_workers,
        shuffle=False,
        batch_size=None,
        pin_memory=True,
    )
    # ddpの使用対応するためのコード
    num_batches = train_dataset_size // (batch_size * gpus)
    train_loader.length = num_batches
    train_loader = train_loader.with_length(num_batches)
    train_loader = train_loader.repeat(nbatches=num_batches)
    train_loader = train_loader.slice(num_batches)

    # ddpの使用対応するためのコード
    num_batches = val_dataset_size // (batch_size * gpus)
    val_loader.length = num_batches
    val_loader = val_loader.with_length(num_batches)
    val_loader = val_loader.repeat(nbatches=num_batches)
    val_loader = val_loader.slice(num_batches)

    return train_loader, val_loader, n_classes
