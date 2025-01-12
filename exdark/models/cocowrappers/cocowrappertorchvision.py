from typing import Optional

from torch import nn, Tensor

from exdark.data.preprocess.labels_storage import coco2coco_like_exdark
from exdark.models.baseexdarkmodel import BaseExDarkModule
from exdark.models.cocowrappers.detection_filter import filter_detections


class COCOWrapperTorchvision(BaseExDarkModule):
    """
    COCOWrapperTorchvision wraps any Torchvison object detection model trained on COCO datasets. After standard
    inference predictions all predictions for categories both present in COCO and ExDark are translated from
    COCO indicates into ExDark indices.
    """

    def __init__(
        self,
        torchvision_detector: nn.Module,
        categories_filter: dict = coco2coco_like_exdark,
        *args,
        **kwargs,
    ):
        super(COCOWrapperTorchvision, self).__init__()
        self.model = torchvision_detector
        self.categories_filter = categories_filter

    def _filter_detections(self, detections: list[dict]) -> list[dict]:
        filtered = filter_detections(detections, self.categories_filter)
        for detection_dict in filtered:
            for key in detection_dict:
                detection_dict[key].to(self.device)
        return filtered

    def forward(
        self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ):
        outputs = self.model(images, targets)
        return self._filter_detections(outputs)
