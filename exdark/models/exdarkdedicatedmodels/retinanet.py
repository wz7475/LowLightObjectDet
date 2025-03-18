from functools import partial

import torch
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from exdark.data.preprocess.labels_storage import exdark_coco_like_labels
from exdark.models.exdarkdedicatedmodels.basedetectortorchvison import (
    BaseDetectorTorchvision,
)


class Retina(BaseDetectorTorchvision):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        num_classes: int = len(exdark_coco_like_labels),
        lr_head: float = 0.005,
        lr_backbone: float = 0.0005,
        freeze_backbone: bool = False,
        use_extended_logging: bool = False,
        use_pretrained_weights: bool = True,
    ):
        super(Retina, self).__init__(
            optimizer,
            scheduler,
            num_classes,
            lr_head,
            lr_backbone,
            freeze_backbone,
            use_extended_logging,
            use_pretrained_weights,
        )

    def _build_model(self, num_classes):
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if self.hparams.use_pretrained_weights else None
        )
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )
        return model
