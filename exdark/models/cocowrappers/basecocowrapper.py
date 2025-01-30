"""
Module for the BaseCOCOWrapper class.
"""

from abc import ABC, abstractmethod

from exdark.models.baseexdarkmodel import BaseExDarkModule
from exdark.models.cocowrappers.detection_filter import filter_detections


class BaseCOCOWrapper(BaseExDarkModule, ABC):
    """
    Base class for COCO wrappers. It provides a common interface for all COCO wrappers.
    """

    @abstractmethod
    def _get_categories_map(self) -> dict:
        pass

    def _filter_detections(self, detections: list[dict]) -> list[dict]:
        filtered = filter_detections(detections, self._get_categories_map())
        for detection_dict in filtered:
            for key in detection_dict:
                detection_dict[key].to(self.device)
        return filtered
