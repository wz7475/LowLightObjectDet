import torch
import torchvision
from lightning import Trainer

from exdark.config import TEST_DIR
from exdark.datamodule import ExDarkDataModule
from exdark.models.cocowrapperfasterrcnn import ExDarkFasterRCNNWrapper

if __name__ == "__main__":
    # model loading
    core_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    wrapped_model = ExDarkFasterRCNNWrapper(core_model)
    wrapped_model.eval()
    print(wrapped_model.device)

    # lightning inference
    torch.set_float32_matmul_precision("high")
    trainer = Trainer(accelerator="gpu")

    exdark_data = ExDarkDataModule(batch_size=2)
    exdark_data.setup_predict_data("data/dataset/split/no_anno")
    preds = trainer.predict(wrapped_model, datamodule=exdark_data)
    print(preds)
