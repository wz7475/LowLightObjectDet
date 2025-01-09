import hydra
from omegaconf import DictConfig

from exdark.data.datamodules.exdarkdatamodule import ExDarkDataModule
from exdark.visulisation.bbox import draw_bbox_from_targets


class TooManySamplesError(Exception):
    pass


@hydra.main(config_path="../../configs", config_name="visualizedata", version_base="1.3")
def main(cfg: DictConfig):
    datamodule: ExDarkDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.batch_size = 1
    num_samples_to_visualize = cfg.num_samples_to_visualize
    test_dataloader = datamodule.test_dataloader()
    data_iter = iter(test_dataloader)

    if num_samples_to_visualize > len(test_dataloader):
        raise TooManySamplesError("Not enough samples to visualize")

    shown_images = 0
    while shown_images < num_samples_to_visualize:
        images, targets = next(data_iter)
        draw_bbox_from_targets(images[0], targets[0])
        shown_images += 1

if __name__ == "__main__":
    main()