**SSD VGG16**
model details
- [pytorch src](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html#torchvision.models.detection.SSD300_VGG16_Weights)
- ssd300_vgg16, SSD300_VGG16_Weights.COCO_V1
- no fine-tuning
data details
-  no preprocessing
results
- mAP_50: 52.543
- mAP_50_95: 26.683
results on new label system
- mAP_50: 52.501
- mAP_50_95: 26.657

**FASTER R-CNN V2**
model details
- [pytorch src](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2)
- fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
- no fine-tuning
data details
- no preprocessing
results
mAP_50: 61.989
mAP_50_95: 32.375


**FASTER R-CNN V1**
model details
- [pytorch src](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2)
- fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN
- no fine-tuning
data details
- no preprocessing
results on old label system
- mAP_50: 54.939
- mAP_50_95: 27.057
results on new label system
- mAP_50: 54.939
- mAP_50_95: 27.057


## new label system
model predicts labels from 0 to 91, they are mapped to ExDark 0-12, additional labels are mapped to class representing other objects - 13