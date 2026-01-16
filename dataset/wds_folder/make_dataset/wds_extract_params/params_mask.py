import json
import webdataset as wds

import pickle
from .wds_pipeline_interface import WdsPipelineInterface


class MaskWdsPipeline(WdsPipelineInterface):
    def _get_decode_callbacks(self) -> list:
        callbacks = [
            wds.handle_extension("video.pickle", pickle.loads),
            wds.handle_extension("stats.json", json.loads),
            wds.handle_extension("person_mask.pickle", pickle.loads),
            wds.handle_extension("obj_mask.pickle", pickle.loads),
            wds.handle_extension("person_bbox.pickle", pickle.loads),
            wds.handle_extension("obj_bbox.pickle", pickle.loads),
        ]
        return callbacks

    def _get_to_tuple_members(self) -> list:
        members = [
            "video.pickle",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "person_mask.pickle",
            "obj_mask.pickle",
            "person_bbox.pickle",
            "obj_bbox.pickle",
        ]
        return members

    def _get_map_tuple_callbacks(self, decode_video) -> list:
        callbacks = [
            lambda x: decode_video.video_decoder(x, update_transform_random_value=True, update_frame_indices=True),
            lambda x: x["label"],  # label
            lambda x: x["label_text"],  # label text
            lambda x: x["filename"],
            lambda x: decode_video.video_decoder(x, with_frame_sec=False, single_channel=True),
            lambda x: decode_video.video_decoder(x, with_frame_sec=False, single_channel=True),
            lambda x: decode_video.video_decoder(x, with_frame_sec=False, single_channel=True),
            lambda x: decode_video.video_decoder(x, with_frame_sec=False, single_channel=True),
        ]
        return callbacks

    def collate_fn(self, batch):
        ret = {
            "video": batch[0].permute(0, 2, 1, 3, 4),
            "label": batch[1],
            "label_text": batch[2],
            "filename": batch[3],
            "person_mask": batch[4].permute(0, 2, 1, 3, 4),
            "obj_mask": batch[5].permute(0, 2, 1, 3, 4),
            "person_bbox": batch[6].permute(0, 2, 1, 3, 4),
            "obj_bbox": batch[7].permute(0, 2, 1, 3, 4),
        }
        return ret
