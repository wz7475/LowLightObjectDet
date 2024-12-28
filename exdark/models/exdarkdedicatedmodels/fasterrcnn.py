import os

import lightning as L
import torch
import torchvision
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from exdark.data.datamodules.gaussnoisedatamodule import GaussNoiseExDarkDataModule
from exdark.data.preprocess.labels_storage import exdark_coco_like_labels
from exdark.models.exdarkdedicatedmodels.basedetectortorchvison import BaseDetectorTorchvision


class FasterRCNN(BaseDetectorTorchvision):
    """
    Faster R-CNN model based on torchvision implementation. It uses ResNet50 as a backbone. Last layer is replaced with
    FastRCNNPredictor to match the number of classes in the dataset.
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
        super(FasterRCNN, self).__init__(optimizer_class, optimizer_params, scheduler_class, scheduler_params, num_classes, lr_head, lr_backbone)


    def _build_model(self, num_classes):
        # get pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        # replace head for different num of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_channels=in_features, num_classes=num_classes)
        return model



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
