import glob as glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from exdark.config import (
    RESIZE_TO, TRAIN_DIR, BATCH_SIZE, CLASSES_COCO
)
from exdark.custom_utils import collate_fn, get_train_transform, get_valid_transform
from exdark.visulisation.bbox import draw_bbox


class ExDarkDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []

        # Get all the image paths in sorted order.
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.dir_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        image_name = self.all_images[idx]
        image_name = image_name
        image_path = os.path.join(self.dir_path, image_name)

        # Read and preprocess the image.
        image = cv2.imread(image_path).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # Capture the corresponding file for getting the annotations.
        annot_filename = image_name.lower() + ".txt"
        annot_file_path = os.path.join(self.dir_path, annot_filename)

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
                labels.append(self.classes.index(object_info[0]))

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
                    xmax_final = 0
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

        # Apply the image transforms.
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)


def create_train_dataset(DIR):
    train_dataset = ExDarkDataset(
        DIR, RESIZE_TO, RESIZE_TO, CLASSES_COCO, get_train_transform()
    )
    return train_dataset


def create_valid_dataset(DIR):
    valid_dataset = ExDarkDataset(
        DIR, RESIZE_TO, RESIZE_TO, CLASSES_COCO, get_valid_transform()
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


if __name__ == '__main__':
    dataset = ExDarkDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES_COCO
    )

    for image, target in dataset:
        draw_bbox(image, target)
