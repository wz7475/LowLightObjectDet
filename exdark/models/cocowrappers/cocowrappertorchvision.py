from typing import Optional

import lightning as L
import torch
from torch import nn, Tensor
from torchmetrics.detection import MeanAveragePrecision

from exdark.data.preprocess.labels_storage import coco2coco_like_exdark


class COCOWrapperTorchvision(L.LightningModule):
    """
    COCOWrapperTorchvision wraps any Torchvison object detection model trained on COCO datasets. After standard
    inference predictions all predictions for categories both present in COCO and ExDark are translated from
    COCO indicates into ExDark indices.
    """

    def __init__(
        self,
        torchvision_detector: nn.Module,
        categories_filter: dict = coco2coco_like_exdark,
    ):
        super(COCOWrapperTorchvision, self).__init__()
        self.model = torchvision_detector
        self.categories_filter = categories_filter
        self.metric = MeanAveragePrecision()

    def _filter_detections(self, detections: list[dict]) -> list[dict]:
        filtered_detections_list = []
        for detections_dict in detections:
            filtered_detections_dict = {"boxes": [], "labels": [], "scores": []}
            for box, label_tensor, score in zip(
                detections_dict["boxes"],
                detections_dict["labels"],
                detections_dict["scores"],
            ):
                label = label_tensor.item()
                if label not in self.categories_filter:
                    continue
                label = self.categories_filter[label]
                filtered_detections_dict["boxes"].append(box.tolist())
                filtered_detections_dict["labels"].append(label)
                filtered_detections_dict["scores"].append(score)
            filtered_detections_list.append(
                dict((k, torch.tensor(v)) for k, v in filtered_detections_dict.items())
            )
        return filtered_detections_list

    def forward(
        self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ):
        outputs = self.model(images, targets)
        return self._filter_detections(outputs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)
        self.metric.update(preds, targets)

    def on_test_epoch_end(self) -> None:
        mAP = self.metric.compute()
        self.log("test_mAP", mAP["map"], prog_bar=True)
        self.log("test_mAP_50", mAP["map_50"], prog_bar=True)
        self.log("test_mAP_75", mAP["map_75"], prog_bar=True)
        self.metric.reset()
