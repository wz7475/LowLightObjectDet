import torchvision

coco_labels = torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1.meta["categories"]
exdark_labels = ['__background__', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cup', 'dog',
                  'motorbike', 'people', 'table']