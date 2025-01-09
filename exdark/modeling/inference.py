import glob as glob
import os.path
from typing import Literal

import cv2
import hydra
import numpy as np
import torch
from lightning import LightningModule, Trainer
from omegaconf import DictConfig

from exdark.modeling.utils import setup_environment
from exdark.visulisation.bbox import (
    draw_bbox_from_preds,
    preprocess_predictions,
    print_predictions,
)


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths: list[str]):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.tensor(image_input, dtype=torch.float), img_path


def run_inference(
    dataset: torch.utils.data.Dataset,
    model: LightningModule,
    device: Literal["cpu", "gpu"],
    visualize: bool,
    threshold: float,
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    trainer = Trainer(accelerator=device)
    predictions = trainer.predict(model, dataloaders=dataloader)

    for (_, img_path), output in zip(dataloader, predictions):
        image = cv2.imread(img_path[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(img_path)
        boxes, scores, pred_classes = preprocess_predictions(output, threshold)
        if visualize:
            draw_bbox_from_preds(image, boxes, scores, pred_classes)
        else:
            print_predictions(boxes, scores, pred_classes)


def load_model(cfg: DictConfig) -> LightningModule:
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.ckpt_path:
        model = model.__class__.load_from_checkpoint(cfg.ckpt_path, **cfg.model)
    return model


def get_image_paths(input_dir: str) -> list[str]:
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        img_paths += glob.glob(os.path.join(input_dir, ext))
    return img_paths


@hydra.main(config_path="../../configs", config_name="inference", version_base="1.3")
def main(cfg: DictConfig):
    setup_environment(cfg.seed)
    model = load_model(cfg)
    img_paths = get_image_paths(cfg.input)
    dataset = InferenceDataset(img_paths)
    run_inference(dataset, model, cfg.device, cfg.visualize, cfg.threshold)


if __name__ == "__main__":
    main()
