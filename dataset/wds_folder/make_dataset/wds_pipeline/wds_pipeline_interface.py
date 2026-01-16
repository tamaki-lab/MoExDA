import abc

import webdataset as wds


class WdsPipelineInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _get_decode_callbacks(self) -> list:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_to_tuple_members(self) -> list:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_map_tuple_callbacks(self, decode_video) -> list:
        raise NotImplementedError()

    def _get_dataset_pipeline_members(self, video_decoder) -> tuple[list, list, list]:
        return (
            self._get_decode_callbacks(),
            self._get_to_tuple_members(),
            self._get_map_tuple_callbacks(video_decoder),
        )

    def __call__(self, dataset: wds.WebDataset, video_decoder) -> wds.WebDataset:
        d_params, t_params, m_params = self._get_dataset_pipeline_members(video_decoder)
        return (
            dataset
            .decode(*d_params)
            .to_tuple(*t_params)
            .map_tuple(*m_params)
        )

    @abc.abstractmethod
    def collate_fn(self, batch) -> dict:
        raise NotImplementedError()
