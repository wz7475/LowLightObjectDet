from typing import Optional

from PIL import ImageDraw
import torch
import torchvision
from torch import nn, Tensor
import lightning as L
from transformers.image_transforms import to_pil_image

from exdark.data.preprocess.labels_storage import coco2coco_like_exdark, exdark_coco_like_labels
from exdark.data.datamodule import ExDarkDataModule


class ExDarkFasterRCNNWrapper(L.LightningModule):
    def __init__(self, torchvision_detector: nn.Module, categories_filter: dict = coco2coco_like_exdark):
        super(ExDarkFasterRCNNWrapper, self).__init__()
        self.model = torchvision_detector
        self.categories_filter = categories_filter

    def _filter_detections(self, detections: list[dict]) -> list[dict]:
        filtered_detections_list = []
        for detections_dict in detections:
            filtered_detections_dict = {"boxes": [], "labels": [], "scores": []}
            for box, label_tensor, score in zip(
                    detections_dict["boxes"],
                    detections_dict["labels"],
                    detections_dict["scores"],
            ):
                label = label_tensor.item()
                if label not in self.categories_filter:
                    continue
                label = self.categories_filter[label]
                filtered_detections_dict["boxes"].append(box.tolist())
                filtered_detections_dict["labels"].append(label)
                filtered_detections_dict["scores"].append(score)
            filtered_detections_list.append(dict((k, torch.tensor(v)) for k, v in filtered_detections_dict.items()))
        return filtered_detections_list

    def forward(
            self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ):
        outputs = self.model(images, targets)
        return self._filter_detections(outputs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    wrapped_model = ExDarkFasterRCNNWrapper(torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights
    )).to(device).eval()
    data_iter = iter(ExDarkDataModule(batch_size=1).val_dataloader())

    for _ in range(3):
        imgs, targets = next(data_iter)
        img_pil = to_pil_image(imgs[0])
        imgs = [img.to(device) for img in imgs]

        with torch.no_grad():
            results = wrapped_model(imgs)

        print(results)
        """
        [{'boxes': tensor([[228.7298,  70.7155, 391.9681, 219.7094],
        [134.9557,  25.8951, 385.8469, 222.0154],
        [244.1486,   3.4234, 389.3126, 199.0562],
        [289.7344, 110.3468, 387.8669, 216.8933],
        [227.2883,  18.5220, 396.0899, 231.3304]]), 'labels': tensor([ 1,  1, 11,  1, 10]), 'scores': tensor([0.9260, 0.4006, 0.1689, 0.1378, 0.0742])}]
        """
        draw = ImageDraw.Draw(img_pil)
        result = results[0]
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            if score > 0.4:
                x, y, x2, y2 = tuple(box)
                draw.rectangle((x, y, x2, y2), outline="red", width=2)
                draw.text((x, y), exdark_coco_like_labels[label.item()], fill="white")
        img_pil.show()
