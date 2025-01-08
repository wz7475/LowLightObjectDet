import os.path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import lightning as L

from exdark.data.datamodules.exdarkdatamodule import ExDarkDataModule
from exdark.modeling.utils import setup_environment, get_logger
from exdark.models.cocowrappers.cocowrappertransformers import COCOWrapperTransformers


# def test_model(model: nn.Module, device: torch.device, batch_size: int, writer: SummaryWriter):
#     """
#     function to evaluate model independently of its architecture and way of training
#     """
#     # test_loader = GammaBrightenExDarkDataModule(batch_size=batch_size).test_dataloader()
#     # test_loader = HVICIDDataModule(batch_size=batch_size).test_dataloader()
#     test_loader = ExDarkDataModule(batch_size=batch_size).test_dataloader()
#     model.eval()
#     prog_bar = tqdm(test_loader, total=len(test_loader))
#     all_targets = []
#     all_preds = []
#     for i, data in enumerate(prog_bar):
#         images, targets = data
#
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         with torch.no_grad():
#             outputs = model(images, targets)
#
#         for i in range(len(images)):
#             true_dict = dict()
#             preds_dict = dict()
#             true_dict["boxes"] = targets[i]["boxes"].detach().cpu()
#             true_dict["labels"] = targets[i]["labels"].detach().cpu()
#
#             preds_dict["boxes"] = outputs[i]["boxes"].detach().cpu()
#             preds_dict["scores"] = outputs[i]["scores"].detach().cpu()
#             preds_dict["labels"] = outputs[i]["labels"].detach().cpu()
#             all_preds.append(preds_dict)
#             all_targets.append(true_dict)
#
#     metric = MeanAveragePrecision()
#     metric.update(all_preds, all_targets)
#     metric_summary = metric.compute()
#     writer.add_scalars(
#         "mAP", dict((k, v.item()) for k, v in metric_summary.items() if k != "classes")
#     )
#     return metric_summary


@hydra.main(config_path="../../configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    setup_environment(cfg.seed)
    logger = get_logger()

    model = hydra.utils.instantiate(cfg.model)
    exdark_data = hydra.utils.instantiate(cfg.datamodule)
    trainer = L.Trainer(
        accelerator=cfg.device,
        logger=logger,
    )
    test_results = trainer.test(model=model, datamodule=exdark_data, ckpt_path=cfg.ckpt_path)
    # check passing null
    print(test_results)


if __name__ == "__main__":
    main()
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    # print(device)
    # # core_model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
    # #     weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    # # )
    # # model = COCOWrapperTorchvision(core_model)
    # model = COCOWrapperTransformers()
    # # model = ExDarkRTDetrWrapper()
    # # model = FasterRCNN.load_from_checkpoint("exdark/5b16v4xi/checkpoints/epoch=142-step=26741.ckpt")
    # print(model.__class__)
    # model = model.to(device)
    #
    # base_dir = os.path.join("eval_runs", "base")
    # os.makedirs(base_dir, exist_ok=True)
    # with SummaryWriter(log_dir=os.path.join(base_dir, "x")) as writer:
    #     metric_summary = test_model(model, device, 7, writer)
    # print(f"mAP_50: {metric_summary['map_50'] * 100:.3f}")
    # print(f"mAP_50_95: {metric_summary['map'] * 100:.3f}")
    # print(metric_summary)
