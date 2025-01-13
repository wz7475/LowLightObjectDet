from typing import Optional

from torch import nn, Tensor

from exdark.data.preprocess.labels_storage import coco2coco_like_exdark
from exdark.models.baseexdarkmodel import BaseExDarkModule
from exdark.models.cocowrappers.basecocowrapper import BaseCOCOWrapper
from exdark.models.cocowrappers.detection_filter import filter_detections


class COCOWrapperTorchvision(BaseCOCOWrapper):
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

    def _get_categories_map(self) -> dict:
        return self.categories_filter

    def forward(
        self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ):
        outputs = self.model(images, targets)
        return self._filter_detections(outputs)
