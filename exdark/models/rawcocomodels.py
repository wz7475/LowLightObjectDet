import torchvision

from torchvision.models.detection import (
    SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

from exdark.models.fasterrcnn import FasterRCNN


def get_faste_rcnn_resnet50() -> FasterRCNN:
    return fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights)
