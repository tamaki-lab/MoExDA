from io import BytesIO
from pathlib import Path
from tqdm import tqdm
import os
import av
import random
import json
from multiprocessing import Process, Manager
import queue

from utils import bytes2kmg, short_side, MyManager, MyShardWriter
from args import arg_factory


def worker(q, lock, pbar, sink, quality, class_to_idx, pos):

    while True:
        try:
            video_file_path = q.get(timeout=1)
        except queue.Empty:
            return
        if video_file_path is None:
            return

        #
        # open a video file
        #

        try:
            container = av.open(str(video_file_path))
        except Exception as e:
            print(e)
            continue
        if len(container.streams.video) == 0:
            print(f'{video_file_path.name} have no video streams. skip.')
            continue

        video_stream_id = 0  # default
        stream = container.streams.video[video_stream_id]

        if stream.frames > 0:
            n_frames = stream.frames
        else:
            # stream.frames is not available for some codecs
            n_frames = int(float(container.duration)
                           / av.time_base * stream.base_rate)

        #
        # split frames
        #

        jpg_byte_list = []
        frame_sec_list = []
        resize_w, resize_h = short_side(
            w=stream.codec_context.width,
            h=stream.codec_context.height,
            size=args.short_side_size)

        with tqdm(
            container.decode(stream),
            total=n_frames,
            position=pos + 1,
            leave=False,
            mininterval=0.5,
        ) as frame_pbar:
            frame_pbar.set_description(f"worker {pos:02d}")
            for frame in frame_pbar:
                frame_sec_list.append(frame.time)
                img = frame.to_image(width=resize_w,
                                     height=resize_h)
                with BytesIO() as buffer:
                    img.save(buffer,
                             format='JPEG',
                             quality=quality)
                    jpg_byte_list.append(buffer.getvalue())

        #
        # prepare
        #

        category_name = video_file_path.parent.name
        label = class_to_idx[category_name]
        key_str = category_name + '/' + video_file_path.stem

        video_stats_dict = {
            '__key__': key_str,
            'video_id': video_file_path.stem,
            'filename': video_file_path.name,
            'category': category_name,
            'label': label,
            'width': stream.codec_context.width,
            'height': stream.codec_context.height,
            'fps': float(stream.base_rate),
            'n_frames': n_frames,
            'duraion': float(container.duration) / av.time_base,
            'timestamps': frame_sec_list,
        }

        #
        # write
        #

        with lock:
            video_stats_dict['shard'] = sink.get_shards()

            sample_dic = {
                '__key__': key_str,
                'video.pickle': (jpg_byte_list, frame_sec_list),
                'stats.json': json.dumps(video_stats_dict),
            }

            sink.write(sample_dic)
            pbar.update(1)
            pbar.set_postfix_str(
                f"shard {sink.get_shards()}, "
                f"size {bytes2kmg(sink.get_size())}")


def make_shards(args):
    data_info_path = "mimetics_class.json"
    with open(data_info_path, "r", encoding="utf-8") as f:
        ds_info = json.load(f)
        class_to_idx = ds_info["class_to_index"]
        class_list = set(class_to_idx.keys())

    class_folders = [
        p for p in Path(args.dataset_path).glob('*/') if p.is_dir()]
    class_folders = set(class_folders)
    video_file_paths = []
    count = []
    for class_folder in class_folders:
        if class_folder.name in class_list:
            paths = [
                path for path in Path(class_folder).glob('**/*')
                if not path.is_dir()
            ]
            video_file_paths.extend(paths)
            count.append(len(paths))

    if args.shuffle:
        random.shuffle(video_file_paths)
    n_samples = len(video_file_paths)

    # https://github.com/pytorch/vision/blob/a8bde78130fd8c956780d85693d0f51912013732/torchvision/datasets/folder.py#L36
    # class_list = sorted(
    #     entry.name for entry in os.scandir(args.dataset_path)
    #     if entry.is_dir())
    # class_to_idx = {cls_name: i for i, cls_name in enumerate(class_list)}

    shard_dir_path = Path(args.shard_path)
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / f'{args.shard_prefix}-%05d.tar')

    # https://qiita.com/tttamaki/items/96b65e6555f9d255ffd9
    MyManager.register('Tqdm', tqdm)
    MyManager.register('Sink', MyShardWriter)

    with MyManager() as my_manager, \
            Manager() as manager:

        #
        # prepare manager objects
        #
        q = manager.Queue()
        lock = manager.Lock()
        pbar = my_manager.Tqdm(
            total=n_samples,
            position=0,
        )
        pbar.set_description("Main process")
        sink = my_manager.Sink(
            pattern=shard_filename,
            maxsize=int(args.max_size_gb * 1000**3),
            maxcount=args.max_count)

        #
        # start workers
        #
        p_all = [Process(target=worker,
                         args=(q, lock, pbar, sink,
                               args.quality, class_to_idx, i))
                 for i in range(args.num_workers)]
        [p.start() for p in p_all]

        for item in video_file_paths:
            q.put(item)
        for _ in range(args.num_workers):
            q.put(None)

        #
        # wait workers, then close
        #
        [p.join() for p in p_all]
        [p.close() for p in p_all]

        dataset_size_filename = str(
            shard_dir_path / f'{args.shard_prefix}-dataset-size.json')
        with open(dataset_size_filename, 'w') as fp:
            json.dump({
                "dataset size": sink.get_counter(),
                "n_classes": len(class_list),
            }, fp)

        sink.close()
        pbar.close()


if __name__ == '__main__':
    args = arg_factory()
    make_shards(args)
