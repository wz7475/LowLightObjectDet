import hydra
from omegaconf import DictConfig

from exdark.data.datamodules.exdarkdatamodule import ExDarkDataModule
from exdark.visulisation.bbox import draw_bbox_from_targets


class TooManySamplesError(Exception):
    pass

def visualize_data(datamodule, num_samples_to_visualize):
    datamodule.batch_size = 1
    test_dataloader = datamodule.test_dataloader()
    data_iter = iter(test_dataloader)

    if num_samples_to_visualize > len(test_dataloader):
        raise TooManySamplesError("Not enough samples to visualize")

    shown_images = 0
    while shown_images < num_samples_to_visualize:
        images, targets = next(data_iter)
        draw_bbox_from_targets(images[0], targets[0])
        shown_images += 1

@hydra.main(
    config_path="../../configs", config_name="visualizedata", version_base="1.3"
)
def main(cfg: DictConfig):
    try:
        datamodule: ExDarkDataModule = hydra.utils.instantiate(cfg.datamodule)
        visualize_data(datamodule, cfg.num_samples_to_visualize)
    except TooManySamplesError as e:
        print(f"Visualization error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
