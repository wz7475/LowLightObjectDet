import argparse
import glob as glob

import cv2
import numpy as np
import torch

from exdark.config import (
    DEVICE
)
from exdark.model import create_sdd300_vgg16_model
from exdark.visulisation.bbox import draw_bbox_from_preds


def run_inference(img_paths: list[str], model: torch.nn.Module, confidence_thr: float = 0.5):
    for i in range(len(img_paths)):
        # image reading
        image = cv2.imread(img_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image_input = torch.tensor(image_input, dtype=torch.float).to(DEVICE)
        image_input = torch.unsqueeze(image_input, 0)
        # Predictions
        with torch.no_grad():
            outputs = model(image_input.to(DEVICE))
        # visualisation
        draw_bbox_from_preds(image, outputs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        help='path to input image directory',
    )
    args = vars(parser.parse_args())
    model = create_sdd300_vgg16_model()
    model.eval()
    model = model.to(DEVICE)
    test_images = glob.glob(f"{args['input']}/*.jpg")

    run_inference(test_images, model)
