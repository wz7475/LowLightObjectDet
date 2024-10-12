import os
import urllib

import torch
import torchvision
from lightning import Trainer

from exdark.datasets import create_valid_dataset, create_valid_loader
from exdark.models.cocowraper import ExDarkAsCOCOWrapper

if __name__ == "__main__":
    # dataloader
    test_dataset = create_valid_dataset(
        'data/dataset/split/tiny',
    )
    print(len(test_dataset))
    test_loader = create_valid_loader(test_dataset, num_workers=15)

    # model loading
    core_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    wrapped_model = ExDarkAsCOCOWrapper(core_model)
    wrapped_model.eval()
    print(wrapped_model.device)

    # lightning inference
    torch.set_float32_matmul_precision("high")
    trainer = Trainer(accelerator="gpu")
    preds = trainer.predict(wrapped_model, test_loader)
    print(preds)
