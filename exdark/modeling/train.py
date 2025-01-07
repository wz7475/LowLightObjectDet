import os

import hydra
import lightning as L
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from exdark.logging.callbacks import (
    LogDataModuleCallback,
    LogModelCallback,
    LogTransformationCallback,
)


def setup_environment(seed: int):
    load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    L.seed_everything(seed)


def get_callbacks():
    checkpoints = ModelCheckpoint(
        save_top_k=1,
        mode="max",
        monitor="val_mAP",
        every_n_epochs=1,
    )
    early_stopping = EarlyStopping(monitor="val_mAP", mode="max", min_delta=0.01, patience=10)
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
    return WandbLogger(project="exdark", save_dir="wandb_artifacts_detrs")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    setup_environment(cfg.seed)
    callbacks = get_callbacks()
    wandb_logger = get_logger()
    model = hydra.utils.instantiate(cfg.model)
    exdark_data = hydra.utils.instantiate(cfg.datamodule)
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
    )
    trainer.fit(model, datamodule=exdark_data)
    test_results = trainer.test(model=model, datamodule=exdark_data, ckpt_path="best")
    print(test_results)

if __name__ == "__main__":
    # python  exdark/modeling/train.py experiment=exp1
    # python  exdark/modeling/train.py
    main()
