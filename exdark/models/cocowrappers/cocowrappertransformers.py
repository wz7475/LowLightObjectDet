from typing import Optional

import torch
from torch import Tensor
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)

from exdark.models.cocowrappers.basecocowrapper import BaseCOCOWrapper
from exdark.models.cocowrappers.detection_filter import filter_detections


class COCOWrapperTransformers(BaseCOCOWrapper):
    """
    COCOWrapperTransformers wraps any Transformers object detection model trained on COCO datasets. After standard
    inference predictions all predictions for categories both present in COCO and ExDark are translated from
    COCO indicates into ExDark indices.
    """

    def __init__(
        self,
        transformers_detector_tag: str = "SenseTime/deformable-detr",
        post_processing_confidence_thr: float = 0.5,
        *args,
        **kwargs,
    ):
        super(COCOWrapperTransformers, self).__init__()
        self.model = AutoModelForObjectDetection.from_pretrained(transformers_detector_tag)
        self.image_processor = AutoImageProcessor.from_pretrained(
            transformers_detector_tag, do_rescale=False
        )
        self.post_processing_confidence_thr = post_processing_confidence_thr

    def _get_categories_map(self) -> dict:
        exdark_categories = [
            "bicycle",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cup",
            "dog",
            "motorcycle",
            "person",
            "dining table",
        ]
        coco_categories = list(self.model.config.id2label.values())
        return {
            coco_categories.index(category_name): idx + 1
            for idx, category_name in enumerate(exdark_categories)
        }


    def forward(self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None):
        input_encoding = self.image_processor(images, return_tensors="pt")
        input_encoding = {k: v.to(self.device) for k, v in input_encoding.items()}
        output_encoding = self.model(**input_encoding)
        outputs = self.image_processor.post_process_object_detection(
            output_encoding,
            target_sizes=torch.tensor([img.shape[1:] for img in images]),
            threshold=self.post_processing_confidence_thr,
        )
        return self._filter_detections(outputs)

