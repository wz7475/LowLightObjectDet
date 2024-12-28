"""
FasterRCNN model:
- frozen backbone
- new classification head trained on exdark
"""

import os
from typing import Optional

import lightning as L
import torch
import torchvision
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.detection import MeanAveragePrecision

from data.labels_storage import exdark_coco_like_labels
from exdark.data.datamodule import ExDarkDataModule, BrightenExDarkDataModule, GammaBrightenExDarkDataModule, \
    GaussNoiseExDarkDataModule


class FasterRCNN(L.LightningModule):
    def __init__(self, num_classes=(len(exdark_coco_like_labels))):
        super(FasterRCNN, self).__init__()
        self.weight_decay = 0.005
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
        return model

    def forward(
            self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ):
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
        self.log("val_mAP", mAP['map'], prog_bar=True)
        self.log("val_mAP_50", mAP['map_50'], prog_bar=True)
        self.log("val_mAP_75", mAP['map_75'], prog_bar=True)
        self.metric.reset()

    def on_fit_start(self) -> None:
        datamodule = self.trainer.datamodule
        self.logger.log_hyperparams({
            "datamodule_name": datamodule.__class__.__name__,
            "batch_size": datamodule.batch_size,
            "dataset_size": len(datamodule.train_dataset)
        })
        print("logged params")

    def configure_optimizers(self):
        lr = 0.005
        params = []
        params.append({
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": lr / 10,
        })
        params.append({
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": lr,
        })
        params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=self.weight_decay)
        lr_scheduler = MultiStepLR(optimizer, milestones=[15,30], gamma=0.1)
        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    # data
    load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    batch_size = 16
    exdark_data = GaussNoiseExDarkDataModule(batch_size=batch_size)
    # checkpoints
    checkpoints = ModelCheckpoint(
        save_top_k=1,
        mode="max",
        monitor="val_mAP",
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
    load_dotenv()
    wandb.login(key=os.environ["WANDB_TOKEN"])
    wandb_logger = WandbLogger(project='exdark')
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["datamodule"] = "GaussNoiseExDarkDataModule"
    wandb_logger.experiment.config["model"] = "fasterrcnn_resnet50_fpn_v2"
    wandb_logger.experiment.config["augmentations"] = """A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(var_limit=(7500, 8000)),
    """
    # TODO: add datamodule specs logging
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["lr_same_for_backbone_and_head"] = False
    model = FasterRCNN()
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=150,
        callbacks=[checkpoints, lr_monitor],
        logger=wandb_logger
    )
    trainer.fit(model, datamodule=exdark_data)
