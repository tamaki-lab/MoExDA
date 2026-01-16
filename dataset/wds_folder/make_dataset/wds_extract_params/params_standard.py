import json
import webdataset as wds
import pickle

from .wds_pipeline_interface import WdsPipelineInterface


class StandardWdsPipeline(WdsPipelineInterface):
    def _get_decode_callbacks(self) -> list:
        callbacks = [
            wds.handle_extension("video.pickle", pickle.loads),
            wds.handle_extension("stats.json", json.loads),
        ]
        return callbacks

    def _get_to_tuple_members(self) -> list:
        members = [
            "video.pickle",
            "stats.json",
            "stats.json",
            "stats.json",
        ]
        return members

    def _get_map_tuple_callbacks(self, decode_video) -> list:
        callbacks = [
            decode_video.video_decoder,
            lambda x: x["label"],  # label
            lambda x: x["category"],  # label text
            lambda x: x["filename"],
        ]
        return callbacks

    def collate_fn(self, batch):
        ret = {
            "video": batch[0],
            "label": batch[1],
            "label_text": batch[2],
            "filename": batch[3],
        }
        return ret
