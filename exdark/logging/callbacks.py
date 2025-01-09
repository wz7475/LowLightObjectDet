from lightning import Callback
from albumentations.core.transforms_interface import ImageOnlyTransform
import pytorch_lightning as pl


class LogTransformationCallback(Callback):
    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        train_transforms = trainer.datamodule.train_transforms
        eval_transforms = trainer.datamodule.eval_transforms
        pl_module.logger.log_hyperparams(
            {
                "train_transforms": self.get_loggable_format(train_transforms),
                "eval_transforms": self.get_loggable_format(eval_transforms),
            }
        )

    def get_loggable_format(self, transforms: list[ImageOnlyTransform]) -> list[dict]:
        transforms_info = []
        for transform in transforms:
            transforms_info.append(
                {
                    "transform_name": transform.__class__.__name__,
                    "params": self.get_explicit_params(transform),
                }
            )
        return transforms_info

    @staticmethod
    def get_explicit_params(transform: ImageOnlyTransform) -> dict:
        """
        Extract explict params set in initialization
        """
        default_transforms = transform.__class__()
        default_transforms_params = (
            default_transforms.get_transform_init_args()
            | default_transforms.get_base_init_args()
        )
        received_transforms_params = (
            transform.get_transform_init_args() | transform.get_base_init_args()
        )
        explicit_params = {}
        for param, value in received_transforms_params.items():
            default_value = default_transforms_params.get(param)
            if default_value != value:
                explicit_params[param] = value
        return explicit_params


class LogDataModuleCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        datamodule = trainer.datamodule
        pl_module.logger.log_hyperparams(
            {
                "datamodule_name": datamodule.__class__.__name__,
                "batch_size": datamodule.batch_size,
                "dataset_size": len(datamodule.train_dataset),
            }
        )


class LogModelCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        pl_module.logger.log_hyperparams({"model": pl_module.model.__class__.__name__})
