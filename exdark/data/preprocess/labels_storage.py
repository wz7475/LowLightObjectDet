import torchvision

coco_labels = torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1.meta["categories"]
exdark_custon_labels = ['__background__', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cup', 'dog',
                        'motorbike', 'people', 'table']
coco2exdark_exceptions = {
    "motorcycle": "motorbike",
    "person": "people",
    "dining table": "table"
}
exdark2coco_exceptions = dict((v, k) for k, v in coco2exdark_exceptions.items())

exdark_coco_like_labels = [exdark2coco_exceptions.get(label, label) for label in exdark_custon_labels]
exdark_idx2label = dict((idx, label) for idx, label in enumerate(exdark_coco_like_labels))
exdark_label2idx = dict((v, k) for k, v in exdark_idx2label.items())

coco2coco_like_exdark = dict(
    (coco_labels.index(category_name), idx) for idx, category_name in enumerate(exdark_coco_like_labels)
)
