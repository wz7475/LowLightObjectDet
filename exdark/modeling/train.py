import os

import lightning as L
import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from exdark.data.datamodules.exdarkdatamodule import ExDarkDataModule
from exdark.data.datamodules.gaussnoisedatamodule import GaussNoiseExDarkDataModule
from exdark.models.exdarkdedicatedmodels.fasterrcnn import FasterRCNN
from exdark.models.callbacks import (
    LogDataModuleCallback,
    LogModelCallback,
    LogTransformationCallback,
)

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
    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)

    # logger
    wandb.login(key=os.environ["WANDB_TOKEN"])
    wandb_logger = WandbLogger(project="exdark")

    # model
    # optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=self.weight_decay)
    # lr_scheduler = MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1))
    optimzer_class = torch.optim.SGD
    optimizer_params = {"momentum": 0.9, "weight_decay": 0.0005}
    scheduler_class = torch.optim.lr_scheduler.MultiStepLR
    scheduler_params = {"milestones": [15, 30], "gamma": 0.1}
    model = FasterRCNN(
        optimzer_class,
        optimizer_params,
        scheduler_class,
        scheduler_params,
        lr_head=0.005,
        lr_backbone=0.0005,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=150,
        callbacks=[
            checkpoints,
            lr_monitor,
            LogDataModuleCallback(),
            LogModelCallback(),
            LogTransformationCallback(),
        ],
        logger=wandb_logger,
    )
    trainer.fit(model, datamodule=exdark_data)
