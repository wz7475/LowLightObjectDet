import torch
import torchvision

RESIZE_TO = 640  # Resize the image for training and transforms.


CLASSES_COCO = torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1.meta["categories"]

# 12 categories + background
NUM_CLASSES_EXDARK = 12 + 1
