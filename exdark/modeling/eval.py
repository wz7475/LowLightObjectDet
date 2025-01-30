"""
This script is used to evaluate the model on the test set. It uses the hydra framework to load the configuration and instantiate the model, datamodule, and logger.
The results are logged to logger and printed to the console.
"""

import hydra
import lightning as L
from omegaconf import DictConfig

from exdark.modeling.utils import setup_environment


@hydra.main(config_path="../../configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    setup_environment(cfg.seed)
    logger = hydra.utils.instantiate(cfg.logger)
    model = hydra.utils.instantiate(
        cfg.model, use_extended_logging=logger.supports_extended_logging
    )
    exdark_data = hydra.utils.instantiate(cfg.datamodule)
    trainer = L.Trainer(
        accelerator=cfg.device,
        logger=logger,
    )
    test_results = trainer.test(
        model=model, datamodule=exdark_data, ckpt_path=cfg.ckpt_path
    )
    print(test_results)


if __name__ == "__main__":
    main()
