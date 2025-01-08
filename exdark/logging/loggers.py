import os

from pytorch_lightning.loggers import WandbLogger, CSVLogger, Logger
import wandb

class ExDarkLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def supports_extended_logging(self) -> bool:
        raise NotImplementedError()


class ExDarkWandBLogger(WandbLogger, ExDarkLogger):
    def __init__(self, token: str, *args, **kwargs):
        wandb.login(key=token)
        super().__init__(*args, **kwargs)

    @property
    def supports_extended_logging(self) -> bool:
        return True


class ExDarkCSVLogger(CSVLogger, ExDarkLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def supports_extended_logging(self) -> bool:
        return False
