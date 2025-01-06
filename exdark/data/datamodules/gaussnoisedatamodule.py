"""
GaussNoiseExDarkDataModule is derived from ExDarkDataModule - training and evaluation data is corrupted with Gaussian Noise.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from exdark.data.datamodules.exdarkdatamodule import ExDarkDataModule


class GaussNoiseExDarkDataModule(ExDarkDataModule):
    """
    GaussNoiseExDarkDataModule is derived from ExDarkDataModule - training and evaluation data is corrupted with Gaussian Noise.
    """

    def __init__(self, batch_size: int, limit_to_n_samples: int | None = None, use_augmentations: bool = True):
        super().__init__(batch_size, limit_to_n_samples, use_augmentations)

    @staticmethod
    def get_train_transformations() -> A.Compose:
        return A.Compose(
            [
                A.Perspective(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(var_limit=(4000, 5000), p=1),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    @staticmethod
    def get_eval_transformations() -> A.Compose:
        return A.Compose(
            [A.GaussNoise(var_limit=(4000, 5000), p=1), ToTensorV2(p=1.0)],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )
