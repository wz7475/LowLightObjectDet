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
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

from data.labels_storage import exdark_coco_like_labels
from exdark.datamodule import ExDarkDataModule


class RTDetr(LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        # TODO: adjust num labels
        super().__init__()
        self.model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd",
                                                                 # num_labels=num_classes,
                                                                 id2label={id: label for id, label in
                                                                           enumerate(exdark_coco_like_labels)},
                                                                 label2id={label: id for id, label in
                                                                           enumerate(exdark_coco_like_labels)},
                                                                 num_queries=300,
                                                                 anchor_image_size=None,
                                                                 ignore_mismatched_sizes=True)
        self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd", do_rescale=False)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.metric = MeanAveragePrecision(iou_thresholds=[0.5])

    @staticmethod
    def _transform_targets_to_transformers_format(targets: list[dict[str, Tensor]]):
        for target in targets:
            target["class_labels"] = target["labels"]
            # del target["labels"]
        return targets

    def forward(self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None):
        """example target
        {'size': tensor([640, 640]), 'image_id': tensor([20]), 'class_labels': tensor([0, 0, 0, 0, 0, 0, 0]), 'boxes': tensor([[0.5146, 0.2154, 0.1104, 0.1758],
        [0.5364, 0.3516, 0.0972, 0.1524],
        [0.4314, 0.3755, 0.1011, 0.1766],
        [0.4360, 0.3524, 0.1055, 0.1553],
        [0.3325, 0.4088, 0.1016, 0.1597],
        [0.5652, 0.5154, 0.1030, 0.1634],
        [0.4436, 0.5579, 0.0981, 0.1619]]), 'area': tensor([412.3077, 241.7583, 204.3956, 500.6594, 327.1795, 117.8022, 140.6593]), 'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0]), 'orig_size': tensor([1365, 2048])}
        """
        targets_for_processor = None
        if targets:
            targets = self._transform_targets_to_transformers_format(targets)
            targets_for_processor = []
            for target in targets:
                targets_for_processor.append({
                    'image_id': target['image_id'].tolist(),
                    'annotations': [
                        {
                            'bbox': box.tolist(),
                            'category_id': label.item(),
                            'area': area.item(),
                            'iscrowd': iscrowd.item()
                        }
                        for box, label, area, iscrowd in zip(
                            target['boxes'],
                            target['labels'],
                            target['area'],
                            target['iscrowd']
                        )
                    ]
                })
        encoding = self.processor(images=images, annotations=targets_for_processor, return_tensors="pt")
        labels = [{k: torch.tensor(v).to(self.device) for k, v in t.items()} for t in targets_for_processor]
        return self.model(pixel_values=encoding["pixel_values"], labels=labels)

    def common_step(self, batch, batch_idx):
        images, targets = batch
        return self.forward(images, targets)

    def training_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)
        loss, loss_dict = outputs.loss, outputs.loss_dict
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.common_step(batch, batch_idx)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        # Process predictions
        preds = self.processor.post_process_object_detection(outputs,
                                                             target_sizes=[(640, 640) for _ in range(len(targets))])
        print(preds)

        # Convert predictions to the format expected by MeanAveragePrecision
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        target_boxes = []
        target_labels = []

        for pred, target in zip(preds, targets):
            pred_boxes.append(pred['boxes'])
            pred_scores.append(pred['scores'])
            pred_labels.append(pred['labels'])
            target_boxes.append(target['boxes'])
            target_labels.append(target['class_labels'])

        self.metric.update(
            preds=[{
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
            } for boxes, scores, labels in zip(pred_boxes, pred_scores, pred_labels)],
            target=[{
                'boxes': boxes,
                'labels': labels,
            } for boxes, labels in zip(target_boxes, target_labels)]
        )
        """sample pred format
        {'boxes': tensor([[0.4309, 0.5275, 0.4492, 0.5810]]), 'labels': tensor([1]), 'scores': tensor([0.5379])}
        """
        return loss

    def on_validation_epoch_end(self) -> None:
        mAP = self.metric.compute()
        self.log("val_mAP", mAP['map'], prog_bar=True)
        self.log("val_mAP_50", mAP['map_50'], prog_bar=True)
        self.log("val_mAP_75", mAP['map_75'], prog_bar=True)
        self.metric.reset()

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
    batch_size = 1
    exdark_data = ExDarkDataModule(batch_size)

    # model
    model = RTDetr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-5)

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
