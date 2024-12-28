from typing import Optional

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision

from exdark.data.preprocess.labels_storage import exdark_coco_like_labels


class BaseDetectorTorchvision(L.LightningModule):
    """
    Base class for all detectors that use torchvision models. It provides a common interface for training and evaluation.
    """

    def __init__(
        self,
        optimizer_class: torch.optim.Optimizer,
        optimizer_params: dict | None,
        scheduler_class: torch.optim.lr_scheduler.LRScheduler | None,
        scheduler_params: dict | None,
        num_classes: int = (len(exdark_coco_like_labels)),
        lr_head: float = 0.005,
        lr_backbone: float = 0.0005,
    ):
        super(BaseDetectorTorchvision, self).__init__()
        self.save_hyperparameters()

        self.model = self._build_model(num_classes)
        self.metric = MeanAveragePrecision()

    def _build_model(self, num_classes: int):
        raise NotImplementedError("Child classes must implement _build_model().")

    def forward(self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None):
        return self.model(images, targets)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])

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
        params = []
        params.append(
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n],
                "lr": self.hparams.lr_backbone,
            }
        )
        params.append(
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" not in n],
                "lr": self.hparams.lr_head,
            }
        )
        params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = self.hparams.optimizer_class(params, **self.hparams.optimizer_params)
        if self.hparams.scheduler_class is None:
            return optimizer
        scheduler = self.hparams.scheduler_class(optimizer, **self.hparams.scheduler_params)
        return [optimizer], [scheduler]
