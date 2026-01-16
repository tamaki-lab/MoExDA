import numpy as np
import random
import warnings


def uniform_clip_sampler(frame_sec_list, clip_sampler_args):
    n_frames = clip_sampler_args["clip_frames"]
    if n_frames <= len(frame_sec_list):
        return np.linspace(0, len(frame_sec_list) - 1, num=n_frames).astype(int)
    else:
        diff = n_frames - len(frame_sec_list)
        is_even = diff % 2 == 0
        pre = np.zeros(diff // 2 if is_even else diff // 2 + 1, dtype=np.int8)
        mid = np.array(range(len(frame_sec_list)))
        suf = np.ones((diff // 2), dtype=np.int8) * (len(frame_sec_list) - 1)
        return np.concatenate([pre, mid, suf])


def center_frame_clip_sampler(frame_sec_list, clip_sampler_args):
    n_frames = clip_sampler_args["clip_frames"]
    if n_frames != 1:
        warnings.warn(
            "when center_frame clip sampler is selected, args.clip_frames is ignored and only one frame is fetched."
        )
    return np.array([len(frame_sec_list) // 2])


def random_start_frame_clip_sampler(frame_sec_list, clip_sampler_args):
    n_frames = clip_sampler_args["clip_frames"]
    if n_frames <= len(frame_sec_list):
        clip_duration = clip_sampler_args["clip_duration"]

        video_duration = frame_sec_list[-1] - frame_sec_list[0]
        max_start_time = video_duration - clip_duration
        start_time = random.uniform(0, max_start_time)
        end_time = start_time + clip_duration
        candidates = [i for i, f in enumerate(frame_sec_list) if start_time <= f <= end_time]
        try:
            frame_indices = np.linspace(candidates[0], candidates[-1], num=n_frames).astype(int)

        except IndexError as e:
            print("catch IndexError:", e)
            print("frame_sec_list:", frame_sec_list)
            print("candidates:", candidates)
        return frame_indices
    else:
        diff = n_frames - len(frame_sec_list)
        is_even = diff % 2 == 0
        pre = np.zeros(diff // 2 if is_even else diff // 2 + 1, dtype=np.int8)
        mid = np.array(range(len(frame_sec_list)))
        suf = np.ones((diff // 2), dtype=np.int8) * (len(frame_sec_list) - 1)
        return np.concatenate([pre, mid, suf])
