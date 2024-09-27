import torch
import torchvision

BATCH_SIZE = 2 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640  # Resize the image for training and transforms.
NUM_EPOCHS = 1  # Number of epochs to train for.
NUM_WORKERS = 1  # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR = 'data/dataset/split/train'
# Validation images and XML files directory.
VALID_DIR = 'data/dataset/split/val'


CLASSES_COCO = torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1.meta["categories"]

# 12 categories + background
NUM_CLASSES_EXDARK = 12 + 1

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'