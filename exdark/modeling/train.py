import os

import hydra
import lightning as L
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from exdark.data.datamodules.gammadatamodule import GammaBrightenExDarkDataModule
from exdark.logging.callbacks import (
    LogDataModuleCallback,
    LogModelCallback,
    LogTransformationCallback,
)


def setup_environment():
    load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def get_data_module(batch_size):
    return GammaBrightenExDarkDataModule(batch_size=batch_size)


def get_callbacks():
    checkpoints = ModelCheckpoint(
        save_top_k=1,
        mode="max",
        monitor="val_mAP",
        every_n_epochs=1,
    )
    early_stopping = EarlyStopping(monitor="val_mAP", mode="max", min_delta=0.005)
    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)
    return [
        checkpoints,
        lr_monitor,
        early_stopping,
        LogDataModuleCallback(),
        LogModelCallback(),
        LogTransformationCallback(),
    ]


def get_logger():
    wandb.login(key=os.environ["WANDB_TOKEN"])
    return WandbLogger(project="exdark")


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    setup_environment()
    batch_size = cfg.data.batch_size
    exdark_data = get_data_module(batch_size)
    callbacks = get_callbacks()
    wandb_logger = get_logger()
    model = hydra.utils.instantiate(cfg.model)
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=cfg.trainer.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
    )
    trainer.fit(model, datamodule=exdark_data)
    trainer.test(ckpt_path="best")


if __name__ == "__main__":
    main()
