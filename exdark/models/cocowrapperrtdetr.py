from typing import Optional

import lightning as L
import torch
from PIL import ImageDraw
from torch import Tensor
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, AutoModelForObjectDetection, AutoImageProcessor
from transformers.image_transforms import to_pil_image

from data.labels_storage import exdark_idx2label
from exdark.datamodule import ExDarkDataModule


class ExDarkRTDetrWrapper(L.LightningModule):
    def __init__(self):
        super(ExDarkRTDetrWrapper, self).__init__()
        self.model = AutoModelForObjectDetection.from_pretrained("jozhang97/deta-swin-large")
        self.image_processor = AutoImageProcessor.from_pretrained(
            "jozhang97/deta-swin-large", do_rescale=False
        )
        self.categories_map = self._get_transformers_coco_to_exdark_mapping()

    def _get_transformers_coco_to_exdark_mapping(self):
        exdark_categories = [
            "bicycle",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cup",
            "dog",
            "motorcycle",
            "person",
            "dining table",
        ]
        coco_categories = list(self.model.config.id2label.values())
        return {
            coco_categories.index(category_name): idx + 1
            for idx, category_name in enumerate(exdark_categories)
        }

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
                if label not in self.categories_map:
                    continue
                label = self.categories_map[label]
                filtered_detections_dict["boxes"].append(box.tolist())
                filtered_detections_dict["labels"].append(label)
                filtered_detections_dict["scores"].append(score)
            filtered_detections_list.append(
                dict((k, torch.tensor(v)) for k, v in filtered_detections_dict.items())
            )
        return filtered_detections_list

    def forward(self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None):
        input_encoding = self.image_processor(images, return_tensors="pt")
        input_encoding = {k: v.to(self.device) for k, v in input_encoding.items()}
        output_encoding = self.model(**input_encoding)
        outputs = self.image_processor.post_process_object_detection(
            output_encoding,
            target_sizes=torch.tensor([img.shape[1:] for img in images]),
            threshold=0.4,
        )
        return self._filter_detections(outputs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(device)
    wrapped_model = ExDarkRTDetrWrapper().to(device)
    data_iter = iter(ExDarkDataModule(batch_size=1).val_dataloader())

    for _ in range(3):
        imgs, targets = next(data_iter)
        img_pil = to_pil_image(imgs[0])

        with torch.no_grad():
            results = wrapped_model(imgs)

        print(results)
        draw = ImageDraw.Draw(img_pil)
        result = results[0]
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=2)
            draw.text((x, y), exdark_idx2label[label.item()], fill="white")

        img_pil.show()
