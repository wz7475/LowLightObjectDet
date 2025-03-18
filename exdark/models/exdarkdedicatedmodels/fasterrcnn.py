import torch
import torchvision

from exdark.data.preprocess.labels_storage import exdark_coco_like_labels
from exdark.models.exdarkdedicatedmodels.basedetectortorchvison import (
    BaseDetectorTorchvision,
)


class FasterRCNN(BaseDetectorTorchvision):
    """
    Faster R-CNN model based on torchvision implementation. It uses ResNet50 as a backbone. Last layer is replaced with
    FastRCNNPredictor to match the number of classes in the dataset.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        num_classes: int = (len(exdark_coco_like_labels)),
        lr_head: float = 0.005,
        lr_backbone: float = 0.0005,
        freeze_backbone: bool = False,
        use_extended_logging: bool = False,
        use_pretrained_weights: bool = True,
    ):
        super(FasterRCNN, self).__init__(
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
        # get pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            if self.hparams.use_pretrained_weights
            else None
        )
        # replace head for different num of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_channels=in_features, num_classes=num_classes
            )
        )
        return model
