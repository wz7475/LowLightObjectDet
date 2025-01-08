from functools import partial

import torch
import torchvision
from torchvision.models.detection.fcos import FCOSClassificationHead

from exdark.data.preprocess.labels_storage import exdark_coco_like_labels
from exdark.models.exdarkdedicatedmodels.basedetectortorchvison import BaseDetectorTorchvision


class Fcos(BaseDetectorTorchvision):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        num_classes: int = len(exdark_coco_like_labels),
        lr_head: float = 0.005,
        lr_backbone: float = 0.0005,
        freeze_backbone: bool = False,
        use_extended_logging: bool = False,
    ):
        super(Fcos, self).__init__(
            optimizer,
            scheduler,
            num_classes,
            lr_head,
            lr_backbone,
            freeze_backbone,
            use_extended_logging,
        )

    def _build_model(self, num_classes):
        model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )
        model.transform.min_size = (640,)
        model.transform.max_size = 640
        return model

    # def configure_optimizers(self):
    #     lr = 0.01
    #     params = []
    #     params.append({
    #         "params": [p for n, p in model.named_parameters() if "backbone" in n],
    #         "lr": lr / 10,
    #     })
    #     params.append({
    #         "params": [p for n, p in model.named_parameters() if "backbone" not in n],
    #         "lr": lr,
    #     })
    #     params = [param for param in self.model.parameters() if param.requires_grad]
    #     optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=self.weight_decay)
    #     return [optimizer]
