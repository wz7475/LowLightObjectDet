import torchvision


CLASSES_COCO = torchvision.models.detection.FCOS_ResNet50_FPN_Weights.COCO_V1.meta[
    "categories"
]
NUM_CLASSES_EXDARK = 12 + 1
