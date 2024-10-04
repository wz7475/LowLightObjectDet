import os
import urllib.request

import torch
import torchvision
from torch import nn

from data.labels_storage import coco2coco_like_exdark


class ExDarkAsCOCOWrapper(nn.Module):
    def __init__(self, torchvision_detector: nn.Module, categories_filter: dict = coco2coco_like_exdark):
        super().__init__()
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

    def forward(self, images):
        outputs = self.model(images)
        return self._filter_detections(outputs)


if __name__ == "__main__":
    img_name = "temp.jpg"
    urllib.request.urlretrieve("http://farm6.staticflickr.com/5341/9632894369_e180ee5731_z.jpg", img_name)
    img = torchvision.io.read_image(img_name).float()
    core_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights
    )
    wrapped_model = ExDarkAsCOCOWrapper(core_model)
    wrapped_model.to(torch.device("cpu"))
    wrapped_model.eval()
    result = wrapped_model([img, img])

    print(result)

    os.remove(img_name)
