import torchvision

from torchvision.models.detection import SSD300_VGG16_Weights


def create_sdd300_vgg16_model():
    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.DEFAULT
    )
    return model
