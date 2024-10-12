import torch
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from exdark.config import DEVICE, NUM_WORKERS
from exdark.datasets import create_valid_test_dataset, create_valid_loader
from exdark.model import create_sdd300_vgg16_model
from torch.utils.data import DataLoader

from exdark.models.cocowraper import ExDarkAsCOCOWrapper


def validate(valid_data_loader: DataLoader, model: nn.Module):
    model.eval()
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    all_targets = []
    all_preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images, targets)

        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()

            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            all_preds.append(preds_dict)
            all_targets.append(true_dict)


    metric = MeanAveragePrecision()
    metric.update(all_preds, all_targets)
    metric_summary = metric.compute()
    return metric_summary


if __name__ == '__main__':

    core_model = create_sdd300_vgg16_model()
    model = ExDarkAsCOCOWrapper(core_model)
    model.eval()
    model = model.to(DEVICE)


    test_dataset = create_valid_test_dataset(
        'data/dataset/split/test',
    )
    print(len(test_dataset))
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)

    metric_summary = validate(test_loader, model)
    print(f"mAP_50: {metric_summary['map_50'] * 100:.3f}")
    print(f"mAP_50_95: {metric_summary['map'] * 100:.3f}")
