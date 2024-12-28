"""
FasterRCNN model:
- frozen backbone
- new classification head trained on exdark
"""

import os
from functools import partial
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
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.fcos import FCOSClassificationHead

from exdark.data.preprocess.labels_storage import exdark_coco_like_labels
from exdark.data.datamodule import ExDarkDataModule


class Fcos(L.LightningModule):
    def __init__(self, num_classes=(len(exdark_coco_like_labels))):
        super(Fcos, self).__init__()
        self.weight_decay = 0.005
        self.model = self._get_fcos(num_classes)
        self.metric = MeanAveragePrecision()

    @staticmethod
    def _get_fcos(num_classes):
        # resnet_backbone = torch.nn.Sequential(
        #     *list(torchvision.models.resnet50(ResNet50_Weights.DEFAULT).children())[:-2])
        # resnet_backbone.out_channels = 2048 # checked in figure in paper
        # anchor_generator = AnchorGenerator(
        #     sizes=((8,), (16,), (32,), (64,), (128,)),
        #     aspect_ratios=((1.0,),)
        # )
        # return FCOS(
        #     resnet_backbone,
        #     num_classes=num_classes,
        #     anchor_generator=anchor_generator,
        # )
        model = torchvision.models.detection.fcos_resnet50_fpn(
            weights='DEFAULT'
        )
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        model.transform.min_size = (640,)
        model.transform.max_size = 640
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
        lr = 0.01
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
        return [optimizer]


if __name__ == "__main__":
    # data
    load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    batch_size = 16
    exdark_data = ExDarkDataModule(batch_size=batch_size)
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
    wandb_logger.experiment.config["datamodule"] = "ExDarkDataModule"
    wandb_logger.experiment.config["model"] = "fcos_resnet_as_in_docs"
    wandb_logger.experiment.config["lr_scheduler"] = "none"
    wandb_logger.experiment.config["augmentations"] = """A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),"""
    # TODO: add datamodule specs logging
    wandb_logger.experiment.config["batch_size"] = batch_size
    model = Fcos()
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=150,
        callbacks=[checkpoints, lr_monitor],
        logger=wandb_logger
    )
    trainer.fit(model, datamodule=exdark_data)
