"""
Module for the detection models trained on COCO.
"""

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN,
    FCOS,
    fcos_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
    RetinaNet,
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)


def get_faster_rcnn_resnet50() -> FasterRCNN:
    return fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )


def get_fcos_resnet50() -> FCOS:
    return fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)


def get_retinanet_resnet50() -> RetinaNet:
    return retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
