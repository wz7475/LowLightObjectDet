"""
FasterRCNN model:
- frozen backbone
- new classification head trained on exdark
"""

import os
import urllib.request
from typing import Optional, Any

import lightning as L
import torch
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from data.labels_storage import exdark_coco_like_labels
from exdark.datamodule import ExDarkDataModule


class FasterRCNN(L.LightningModule):
    def __init__(self, num_classes=(len(exdark_coco_like_labels))):
        super(FasterRCNN, self).__init__()
        self.model = self._get_faster_rcnn(num_classes)
        self.metric = MeanAveragePrecision()

    @staticmethod
    def _get_faster_rcnn(num_classes):
        # get pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        # replace head for different num of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_channels=in_features, num_classes=num_classes)
        # freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        return model

    def forward(
            self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ):
        return self.model(images, targets)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch
        preds = self.model(images)
        self.metric.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        mAP = self.metric.compute()
        self.log("val_mAP", mAP['map'], prog_bar=True)
        self.log("val_mAP_50", mAP['map_50'], prog_bar=True)
        self.log("val_mAP_75", mAP['map_75'], prog_bar=True)
        self.metric.reset()

    def configure_optimizers(self):
        params = [param for param in self.model.parameters() if param.requires_grad]
        return torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


if __name__ == "__main__":
    exdark_data = ExDarkDataModule(batch_size=1)

    wrapped_model = FasterRCNN()
    trainer = L.Trainer(accelerator="gpu", max_epochs=20, log_every_n_steps=4)
    trainer.fit(wrapped_model, datamodule=exdark_data)
