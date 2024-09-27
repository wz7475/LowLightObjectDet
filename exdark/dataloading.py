import cv2
import numpy as np
from torch.utils.data import DataLoader

from exdark.config import (
    RESIZE_TO, TRAIN_DIR, BATCH_SIZE, CLASSES_COCO
)
from exdark.custom_utils import collate_fn, get_train_transform, get_valid_transform
from exdark.dataset import ExDarkDataset


# Prepare the final datasets and data loaders.
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


# execute `dataloading.py`` using Python command from
# Terminal to visualize sample images
# USAGE: python dataloading.py
if __name__ == '__main__':
    dataset = ExDarkDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES_COCO
    )
    print(f"Number of training images: {len(dataset)}")


    # function to visualize a single sample
    def visualize_sample(image, target, idx):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES_COCO[target['labels'][box_num]]
            # try:
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # except:
            #     image = cv2.cvtColor(image.detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
            # image = np.array(image)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 0, 255),
                2
            )
            cv2.putText(
                image,
                label,
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()


    NUM_SAMPLES_TO_VISUALIZE = len(dataset)
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(f"Sample {i + 1}: {image.shape}")
        visualize_sample(image, target, i)
