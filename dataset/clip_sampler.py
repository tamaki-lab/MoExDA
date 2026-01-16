import numpy as np
import random


def uniform_clip_sampler(frame_sec_list, clip_sampler_args):
    n_frames = clip_sampler_args['clip_frames']
    return np.linspace(0, len(frame_sec_list) - 1, num=n_frames).astype(int)


def random_clip_sampler(frame_sec_list, clip_sampler_args):
    n_frames = clip_sampler_args['clip_frames']
    clip_duration = clip_sampler_args['clip_duration']

    video_duration = frame_sec_list[-1] - frame_sec_list[0]
    max_start_time = video_duration - clip_duration
    start_time = random.uniform(0, max_start_time)
    end_time = start_time + clip_duration
    candidtes = [i for i, f in enumerate(frame_sec_list)
                 if start_time <= f <= end_time]

    frame_indices = np.linspace(
        candidtes[0], candidtes[-1], num=n_frames).astype(int)
    return frame_indices
