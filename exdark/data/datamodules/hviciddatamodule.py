"""
Module contains HVICIDDataModule which is derived from ExDarkDataModule - training and evaluation data is lightened by
external model HVICID Net.
"""

from exdark.config import TRAIN_DIR_LIGHTEN, RESIZE_TO, VALID_DIR_LIGHTEN, TEST_DIR_LIGHTEN
from exdark.data.datamodules.exdarkdatamodule import ExDarkDataModule
from exdark.data.datasets import ExDarkDataset


class HVICIDDataModule(ExDarkDataModule):
    """
    HVICIDDataModule is derived from ExDarkDataModule - training and evaluation data is lightened by external 
    model HVICID Net (https://github.com/Fediory/HVI-CIDNet).
    """
    def __init__(self, batch_size: int, limit_to_n_samples: int | None = None, use_augmentations: bool=True):
        super().__init__(batch_size, limit_to_n_samples, use_augmentations)
        self.train_dataset = ExDarkDataset(
            TRAIN_DIR_LIGHTEN, RESIZE_TO, RESIZE_TO, self.train_transforms
            # "data/dataset/split/tiny", RESIZE_TO, RESIZE_TO, self.train_transforms
        )
        self.val_dataset = ExDarkDataset(
            VALID_DIR_LIGHTEN, RESIZE_TO, RESIZE_TO, self.eval_transforms
        )
        self.test_dataset = ExDarkDataset(
            TEST_DIR_LIGHTEN, RESIZE_TO, RESIZE_TO, self.eval_transforms
        )
