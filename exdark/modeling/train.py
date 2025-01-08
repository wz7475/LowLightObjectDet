import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from omegaconf import DictConfig

from exdark.logging.callbacks import (
    LogDataModuleCallback,
    LogModelCallback,
    LogTransformationCallback,
)
from exdark.logging.loggers import ExDarkLogger
from exdark.modeling.utils import setup_environment


def get_callbacks(use_extended_logging: bool):
    training_flow_callbacks = [
        EarlyStopping(monitor="val_mAP", mode="max", min_delta=0.01, patience=10),
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        ModelCheckpoint(
            save_top_k=1,
            mode="max",
            monitor="val_mAP",
            every_n_epochs=1,
        ),
    ]
    extended_logging_callbacks = (
        []
        if use_extended_logging
        else [LogDataModuleCallback(), LogModelCallback(), LogTransformationCallback()]
    )
    return training_flow_callbacks + extended_logging_callbacks


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_environment(cfg.seed)
    logger: ExDarkLogger = hydra.utils.instantiate(cfg.logger)
    callbacks = get_callbacks(use_extended_logging=logger.supports_extended_logging)
    model = hydra.utils.instantiate(cfg.model, use_extended_logging=logger.supports_extended_logging)
    exdark_data = hydra.utils.instantiate(cfg.datamodule)
    trainer = L.Trainer(
        accelerator=cfg.device,
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model, datamodule=exdark_data)
    test_results = trainer.test(model=model, datamodule=exdark_data, ckpt_path="best")
    print(test_results)


if __name__ == "__main__":
    main()
