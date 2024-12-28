"""
Module for GammaBrightenExDarkDataModule class which is derived from ExDarkDataModule - training and evaluation data is
gamma brightened.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from exdark.data.datamodules.exdarkdatamodule import ExDarkDataModule


class GammaBrightenExDarkDataModule(ExDarkDataModule):
    """
    GammaBrightenExDarkDataModule is derived from ExDarkDataModule - training and evaluation data is gamma brightened.
    """
    def __init__(self, batch_size: int):
        super().__init__(batch_size)

    @staticmethod
    def get_train_transformations() -> A.Compose:
        return A.Compose([
            A.RandomGamma(gamma_limit=(40, 60), p=1.0),
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    @staticmethod
    def get_eval_transformations() -> A.Compose:
        return A.Compose([
            A.RandomGamma(gamma_limit=(40, 60), p=1.0),
            ToTensorV2(p=1.0)
        ],
            bbox_params={
                'format': 'pascal_voc',
                'label_fields': ['labels']
            })
