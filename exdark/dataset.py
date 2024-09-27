import glob as glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2.functional import get_image_size


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
        # image = cv2.imread(image_path)
        image = Image.open(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image_resized = cv2.resize(image, (self.width, self.height))
        image_resized = image.resize((self.width, self.height))

        # Capture the corresponding file for getting the annotations.
        annot_filename = image_name.lower() + ".txt"
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []

        # Original image width and height.
        image_width = image.size[0]
        image_height = image.size[1]

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

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # Bounding box to tensor.
        for i, box in enumerate(boxes):
            # Clipping the bounding box coordinates to ensure they are within [0, width] and [0, height] for x and y, respectively.
            xmin_clipped = np.clip(box[0], 0, self.width)
            ymin_clipped = np.clip(box[1], 0, self.height)
            xmax_clipped = np.clip(box[2], 0, self.width)
            ymax_clipped = np.clip(box[3], 0, self.height)

            # Update the box with clipped values
            boxes[i] = [xmin_clipped, ymin_clipped, xmax_clipped, ymax_clipped]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Apply the image transforms.
        if self.transforms:
            sample = self.transforms(image=np.array(image)/255.0,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)
