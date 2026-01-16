import json
import webdataset as wds

import pickle

from dataset.wds_folder.actionswap_type import ActionSwapType
from .wds_pipeline_interface import WdsPipelineInterface


class ActionSwapWdsPipeline(WdsPipelineInterface):
    def _get_decode_callbacks(self) -> list:
        callbacks = [
            wds.handle_extension("original_video.pickle", pickle.loads),
            wds.handle_extension("actionswap_video.pickle", pickle.loads),
            wds.handle_extension("person_inpainting_video.pickle", pickle.loads),
            wds.handle_extension("person_only_video.pickle", pickle.loads),
            wds.handle_extension("stats.json", json.loads),
        ]
        return callbacks

    def _get_to_tuple_members(self) -> list:
        members = [
            "original_video.pickle",
            "stats.json",
            "stats.json",
            "stats.json",
            "actionswap_video.pickle",
            "person_inpainting_video.pickle",
            "person_only_video.pickle",
            "stats.json",
            "stats.json",
        ]
        return members

    def _get_map_tuple_callbacks(self, decode_video) -> list:
        callbacks = [
            decode_video.video_decoder,
            lambda x: x["category_id"],
            lambda x: x["category"],
            lambda x: x["filename"],
            decode_video.video_decoder,
            decode_video.video_decoder,
            decode_video.video_decoder,
            lambda x: x["bg_category_id"],
            lambda x: x["bg_category"],
        ]
        return callbacks

    def collate_fn(self, batch):
        ret = {
            ActionSwapType.Original.value: batch[0],
            "label": batch[1],
            "label_text": batch[2],
            "filename": batch[3],
            ActionSwapType.ActionSwap.value: batch[4],
            ActionSwapType.PersonInpainting.value: batch[5],
            ActionSwapType.PersonOnly.value: batch[6],
            "bg_label": batch[7],
            "bg_label_text": batch[8],
        }
        return ret
