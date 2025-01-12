from abc import ABC, abstractmethod
import lightning as L
from torchmetrics.detection import MeanAveragePrecision
from torch import Tensor
from typing import Optional


class BaseExDarkModule(L.LightningModule, ABC):
    """
    Base class for ExDark-based object detectors.
    Declared abstract so it cannot be instantiated directly,
    forcing child classes to implement required methods.
    """

    def __init__(self):
        super().__init__()
        self.metric = MeanAveragePrecision()

    @abstractmethod
    def forward(
        self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ):
        """
        Must be implemented by child classes.
        """
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch[0]
        return self.forward(images)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)
        self.metric.update(preds, targets)

    def on_test_epoch_end(self):
        mAP = self.metric.compute()
        self.log("test_mAP", mAP["map"], prog_bar=True)
        self.log("test_mAP_50", mAP["map_50"], prog_bar=True)
        self.log("test_mAP_75", mAP["map_75"], prog_bar=True)
        self.metric.reset()
