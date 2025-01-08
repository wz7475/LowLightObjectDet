import hydra
import lightning as L
from omegaconf import DictConfig

from exdark.modeling.utils import setup_environment


@hydra.main(config_path="../../configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    setup_environment(cfg.seed)
    logger = hydra.utils.instantiate(cfg.logger)
    model = hydra.utils.instantiate(cfg.model)
    exdark_data = hydra.utils.instantiate(cfg.datamodule)
    trainer = L.Trainer(
        accelerator=cfg.device,
        logger=logger,
    )
    test_results = trainer.test(model=model, datamodule=exdark_data, ckpt_path=cfg.ckpt_path)
    print(test_results)


if __name__ == "__main__":
    main()
