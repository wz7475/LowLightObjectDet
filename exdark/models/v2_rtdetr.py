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
from torchmetrics.detection import MeanAveragePrecision
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor, RTDetrForObjectDetection

from data.labels_storage import exdark_coco_like_labels
from exdark.datamodule import ExDarkDataModule


class RTDetr(L.LightningModule):
    def __init__(
            self,
            num_classes: int,
            lr: float = 0.0001,
            weight_decay: float = 0.0005,
            model_name: str = "PekingU/rtdetr_r50vd"
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model and image processor
        self.model = RTDetrForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            anchor_image_size=None,
        )

        self.image_processor = RTDetrImageProcessor.from_pretrained(
            model_name,
            do_resize=True,
            size={"height": 640, "width": 640},
            do_pad=True
        )

        # Initialize metric
        self.metric = MeanAveragePrecision(box_format='xyxy')

    def _process_batch(self, batch):
        images, targets = batch

        # Process images
        pixel_values = self.image_processor(
            images=images,
            return_tensors="pt"
        )["pixel_values"].to(self.device)

        # Process targets
        processed_targets = []
        for target in targets:
            processed_target = {
                "boxes": target["boxes"],
                "class_labels": target["labels"],
                "image_id": target["image_id"],
                "area": target["area"],
                "iscrowd": target["iscrowd"]
            }
            processed_targets.append(processed_target)

        return pixel_values, processed_targets

    def training_step(self, batch, batch_idx):
        pixel_values, targets = self._process_batch(batch)

        outputs = self.model(
            pixel_values=pixel_values,
            labels=targets
        )

        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, targets = self._process_batch(batch)

        outputs = self.model(pixel_values=pixel_values)
        processed_outputs = self.image_processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=[(640, 640)] * len(batch[0])  # Adjust if using different sizes
        )

        # Convert predictions to format expected by MeanAveragePrecision
        predictions = []
        for output in processed_outputs:
            predictions.append({
                "boxes": output["boxes"],
                "scores": output["scores"],
                "labels": output["labels"]
            })

        # Convert targets to format expected by MeanAveragePrecision
        formatted_targets = []
        for target in targets:
            formatted_targets.append({
                "boxes": target["boxes"],
                "labels": target["class_labels"]
            })

        self.metric.update(predictions, formatted_targets)

    def on_validation_epoch_end(self):
        metrics = self.metric.compute()

        # Log metrics
        self.log("val_mAP", metrics["map"], prog_bar=True)
        self.log("val_mAP_50", metrics["map_50"], prog_bar=True)
        self.log("val_mAP_75", metrics["map_75"], prog_bar=True)

        self.metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr / 100
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mAP"
            }
        }



if __name__ == "__main__":
    # data
    batch_size = 1
    exdark_data = ExDarkDataModule(batch_size)

    # model
    model = RTDetr(len(exdark_coco_like_labels))

    # training specs
    checkpoints = ModelCheckpoint(
        save_top_k=1,
        mode="max",
        monitor="val_loss",
        save_last=True,
        every_n_epochs=1,
    )

    # logging
    LOG = False

    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
    if LOG:
        load_dotenv()
        wandb.login(key=os.environ["WANDB_TOKEN"])
        wandb_logger = WandbLogger(project='exdark')
        # TODO: add datamodule specs logging
        wandb_logger.experiment.config["batch_size"] = batch_size
        # training
        trainer = L.Trainer(
            accelerator="gpu",
            max_epochs=100,
            callbacks=[checkpoints, lr_monitor],
            logger=wandb_logger,
        )
        trainer.fit(model, datamodule=exdark_data)
    else:
        # training
        trainer = L.Trainer(
            accelerator="gpu",
            max_epochs=100,
            callbacks=[checkpoints, lr_monitor],
            # logger=wandb_logger,
        )
        trainer.fit(model, datamodule=exdark_data)
