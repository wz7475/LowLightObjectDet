import torchvision

coco_labels = torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1.meta["categories"]
exdark_labels = ['__background__', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cup', 'dog',
                 'motorbike', 'people', 'table']

coco2exdark_exceptions = {
    "motorcycle": "motorbike",
    "person": "people",
    "dining table": "table"
}

exdark2coco_exceptions = dict((v, k) for k, v in coco2exdark_exceptions.items())
