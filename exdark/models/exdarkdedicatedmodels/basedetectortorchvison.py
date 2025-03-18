from typing import Optional

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision

from exdark.data.preprocess.labels_storage import exdark_coco_like_labels
from exdark.models.baseexdarkmodel import BaseExDarkModule


class BaseDetectorTorchvision(BaseExDarkModule):
    """
    Base class for all detectors that use torchvision models. It provides a common interface for training and evaluation.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        num_classes: int = (len(exdark_coco_like_labels)),
        lr_head: float = 0.005,
        lr_backbone: float = 0.0005,
        freeze_backbone: bool = False,
        use_extended_logging: bool = False,
        use_pretrained_weights: bool = True,
    ):
        super(BaseDetectorTorchvision, self).__init__()
        self.save_hyperparameters(logger=use_extended_logging)

        self.model = self._build_model(num_classes)
        self.metric = MeanAveragePrecision()

    def _build_model(self, num_classes: int):
        raise NotImplementedError("Child classes must implement _build_model().")

    def forward(
        self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch
        loss_dict = self.forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch
        preds = self.model(images)
        self.metric.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        mAP = self.metric.compute()
        self.log("val_mAP", mAP["map"], prog_bar=True)
        self.log("val_mAP_50", mAP["map_50"], prog_bar=True)
        self.log("val_mAP_75", mAP["map_75"], prog_bar=True)
        self.metric.reset()

    def configure_optimizers(self):
        if self.hparams.freeze_backbone:
            for n, p in self.named_parameters():
                if "backbone" in n:
                    p.requires_grad = False
        params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.hparams.lr_backbone,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if "backbone" not in n
                ],
                "lr": self.hparams.lr_head,
            },
        ]
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is None:
            return optimizer
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]
