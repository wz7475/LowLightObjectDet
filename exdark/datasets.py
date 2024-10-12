import glob as glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from exdark.config import (
    RESIZE_TO, TEST_DIR, BATCH_SIZE
)
from exdark.custom_utils import collate_fn, get_train_transform, get_valid_transform
from exdark.visulisation.bbox import draw_bbox_from_targets


class ExDarkDataset(Dataset):
    def __init__(self, dir_path, width, height, transforms=None):
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

    def _get_target(self, image: np.array, annot_file_path: str, idx: int):
        boxes = []
        labels = []

        # Original image width and height.
        image_width = image.shape[1]
        image_height = image.shape[0]

        # read bboxes coordinates in coco style and convert them to pascal voc style
        with open(annot_file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:
                object_info = line.split(",")
                labels.append(int(object_info[0]))

                l = float(object_info[1])
                t = float(object_info[2])
                w = float(object_info[3])
                h = float(object_info[4])
                xmin = l
                ymin = t
                xmax = l + w
                ymax = t + h

                xmin_final = (xmin / image_width) * self.width
                xmax_final = (xmax / image_width) * self.width
                ymin_final = (ymin / image_height) * self.height
                ymax_final = (ymax / image_height) * self.height

                # Check that all coordinates are within the image.
                if xmax_final > self.width:
                    xmax_final = self.width
                if ymax_final > self.height:
                    ymax_final = self.height
                if xmin_final < 0:
                    xmin_final = 0
                if ymin_final < 0:
                    ymax_final = 0

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area  # area can be used to for evaluation to group small, medium and large objects
        target["iscrowd"] = iscrowd  # iscrowd set to True indicates crowd i.e. to ignore this objects
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        return target

    def __getitem__(self, idx):
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


def create_train_dataset(DIR):
    train_dataset = ExDarkDataset(
        DIR, RESIZE_TO, RESIZE_TO, get_train_transform()
    )
    return train_dataset


def create_valid_test_dataset(DIR):
    valid_dataset = ExDarkDataset(
        DIR, RESIZE_TO, RESIZE_TO, get_valid_transform()
    )
    return valid_dataset


def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader


def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return valid_loader

def create_inference_loader(inference_dataset, num_workers=0):
    return DataLoader(
        inference_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )


if __name__ == '__main__':
    dataset = ExDarkDataset(TEST_DIR, RESIZE_TO, RESIZE_TO)

    for image, target in dataset:
        draw_bbox_from_targets(image, target)
