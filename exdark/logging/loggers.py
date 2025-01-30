from pytorch_lightning.loggers import WandbLogger, CSVLogger, Logger
import wandb


class ExDarkLogger(Logger):
    """
    Interface for ExDark loggers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def supports_extended_logging(self) -> bool:
        raise NotImplementedError()


class ExDarkWandBLogger(WandbLogger, ExDarkLogger):
    """
    ExDarkWandBLogger is a logger that logs to Weights and Biases platform. It supports extended logging.
    """

    def __init__(self, token: str, *args, **kwargs):
        wandb.login(key=token)
        super().__init__(*args, **kwargs)

    @property
    def supports_extended_logging(self) -> bool:
        return True


class ExDarkCSVLogger(CSVLogger, ExDarkLogger):
    """
    ExDarkCSVLogger is a logger that logs to CSV files. It does not support extended logging.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def supports_extended_logging(self) -> bool:
        return False
