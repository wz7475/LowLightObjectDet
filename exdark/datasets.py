import glob as glob
import os
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from exdark.config import (
    RESIZE_TO, TEST_DIR
)
from exdark.visulisation.bbox import draw_bbox_from_targets


class ExDarkDataset(Dataset):
    def __init__(self, dir_path, width, height, transforms: A.Compose=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []

        # Get all the image paths in sorted order.
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.dir_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def _get_bbox(self, raw_bbox: list[float], img_width: float, img_height: float) -> list[float]:
        l = raw_bbox[0]
        t = raw_bbox[1]
        a = raw_bbox[2]
        b = raw_bbox[3]
        return [
            max((l / img_width) * self.width, 0),
            max((t/ img_height) * self.height, 0),
            min(((l+a) / img_width) * self.width, self.width),
            min(((t+b) / img_height) * self.height, self.height)
        ]

    def _get_target(self, image: np.array, annot_file_path: str, idx: int):
        boxes = []
        labels = []
        image_width = image.shape[1]
        image_height = image.shape[0]

        # read bboxes coordinates in coco style and convert them to pascal voc style
        with open(annot_file_path, 'r') as file:
            for line in file.readlines():
                object_info = line.strip().split(",")
                labels.append(int(object_info[0]))
                boxes.append(self._get_bbox([float(x) for x in object_info[1:5]], image_width, image_height))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["size"] = torch.tensor([self.height, self.width])
        target["area"] = area  # area can be used to for evaluation to group small, medium and large objects
        target["iscrowd"] = iscrowd  # iscrowd set to True indicates crowd i.e. to ignore this objects
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        return target

    def __getitem__(self, idx):
        """
        return tuple
        - image: ndarrray of shape (640, 640, 3)
        - target: dict like {'area': tensor([121421.4609,  96218.9453]), 'boxes': tensor([[132.0000, 202.1053, 451.0000, 582.7368],
        [314.0000, 232.4211, 604.0000, 564.2105]]), 'image_id': tensor([0]), 'iscrowd': tensor([0, 0]), 'labels': tensor([1, 1]), 'size': tensor([640, 640])}
        """
        # Capture the image name and the full image path.
        image_name = self.all_images[idx]
        image_name = image_name
        image_path = os.path.join(self.dir_path, image_name)

        # Read and preprocess the image.
        image = cv2.imread(image_path).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.height and self.width:
            image_resized = cv2.resize(image, (self.width, self.height))
        else:
            image_resized = image.copy()
        image_resized /= 255.0

        annot_filename = image_name.lower() + ".txt"
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        target = self._get_target(image, annot_file_path, idx) if os.path.exists(annot_file_path) else None

        # Apply the image transforms.
        if self.transforms:
            if target:
                sample = self.transforms(image=image_resized,
                                         bboxes=target['boxes'],
                                         labels=target["labels"])
                target['boxes'] = torch.Tensor(sample['bboxes'])
                image_resized = sample['image']
                if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
            else:
                sample = self.transforms(image=image_resized)
                image_resized = sample['image']
        return image_resized, target

    def __len__(self):
        return len(self.all_images)


if __name__ == '__main__':
    dataset = ExDarkDataset(TEST_DIR, RESIZE_TO, RESIZE_TO)

    for image, target in dataset:
        draw_bbox_from_targets(image, target)
