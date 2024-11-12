import os
from typing import Optional

import lightning as L
import torch
import wandb
from dotenv import load_dotenv
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from transformers import AutoModelForObjectDetection

from data.labels_storage import exdark_coco_like_labels
from exdark.datamodule import ExDarkDataModule


class RTDetr(LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_classes=(len(exdark_coco_like_labels))):
        # TODO: adjust num labels
        super().__init__()
        print(num_classes)
        self.model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd",
                                                              # num_labels=num_classes,
                                                            id2label={id: label for id, label in enumerate(exdark_coco_like_labels)},
                                                            label2id={label: id for id, label in enumerate(exdark_coco_like_labels)},
                                                                 num_queries=300,
                                                              anchor_image_size=None,
                                                              ignore_mismatched_sizes=True)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    @staticmethod
    def _transform_targets(targets: list[dict[str, Tensor]]):
        for target in targets:
            target["class_labels"] = target["labels"]
            del target["labels"]
        return targets

    def forward(self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None):
        pixel_values = torch.stack(images)
        if targets:
            targets = self._transform_targets(targets)
            # try:
            #     return self.model(pixel_values=pixel_values, labels=targets)
            # except RuntimeError as e:
            #     print(e, targets)
            for target in targets:
                try:
                    assert target["boxes"].size(0) == target["class_labels"].size(0), "Mismatched boxes and labels"
                except AssertionError:
                    print(f"boxes: {target['boxes'].size(0)}")
                    print(f"class_labels: {target['class_labels'].size(0)}")
                    raise AssertionError()
            return self.model(pixel_values=pixel_values, labels=targets)
        return self.model(pixel_values=pixel_values)

    def common_step(self, batch, batch_idx):
        images, targets = batch
        return self.forward(images, targets)

    def training_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)
        loss, loss_dict = outputs.loss, outputs.loss_dict
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        # TODO: add mAP logging
        # mAP = self.metric.compute()
        # self.log("val_mAP", mAP['map'], prog_bar=True)
        # self.log("val_mAP_50", mAP['map_50'], prog_bar=True)
        # self.log("val_mAP_75", mAP['map_75'], prog_bar=True)
        # self.metric.reset()
        pass

    def on_fit_start(self) -> None:
        """TODO: log datamodule specs"""

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)
        return optimizer


if __name__ == "__main__":
    # data
    batch_size = 16
    exdark_data = ExDarkDataModule(batch_size)

    # model
    model = RTDetr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

    # training specs
    checkpoints = ModelCheckpoint(
        save_top_k=3,
        mode="max",
        monitor="val_loss",
        save_last=True,
        every_n_epochs=1,
    )

    # logging
    load_dotenv()
    wandb.login(key=os.environ["WANDB_TOKEN"])
    wandb_logger = WandbLogger(project='exdark')
    wandb_logger.experiment.config["batch_size"] = batch_size
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)

    # training
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=100,
        callbacks=[checkpoints, lr_monitor],
        logger=wandb_logger,
    )
    trainer.fit(model, datamodule=exdark_data)
