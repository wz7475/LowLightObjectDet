import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from config import DEVICE, NUM_WORKERS
from datasets import create_valid_dataset, create_valid_loader
from model import create_sdd300_vgg16_model, create_fasterrcnn_v1_model


def map_range_cooc_to_exdark(labels: torch.Tensor) -> torch.Tensor:
    """ quick fix for coco range and gaps - maps labels to 0-12 range
    below labels obtained in ./data/labels_mappers.py
    13 stands for other labels which are detected by models trained on COCO, but not included in ExDark """
    labels_map = {0: 0, 1: 11, 2: 1, 3: 5, 4: 10, 6: 4, 9: 2, 17: 6, 18: 9, 44: 3, 47: 8, 62: 7, 67: 12}
    mapped_labels = []
    for cur_label_tensor in labels:
        cur_label = cur_label_tensor.item()
        if cur_label in labels_map:
            mapped_labels.append(labels_map[cur_label])
        else:
            mapped_labels.append(13)
    return torch.tensor(mapped_labels)


# Evaluation function
def validate(valid_data_loader, model):
    print('Validating')
    model.eval()

    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    max_label = -1
    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            true_dict['labels'] = map_range_cooc_to_exdark(true_dict['labels'])

            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds_dict['labels'] = map_range_cooc_to_exdark(preds_dict['labels'])
            max_label = max(preds_dict['labels']).item() if max(preds_dict['labels']).item() > max_label else max_label

            preds.append(preds_dict)
            target.append(true_dict)
        #####################################
    print(max_label)

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary


if __name__ == '__main__':
    model = create_sdd300_vgg16_model()
    # model = create_fasterrcnn_v2_model()
    # model = create_fasterrcnn_v1_model()

    model = model.to(DEVICE).eval()

    test_dataset = create_valid_dataset(
        'data/dataset/split/test',
    )
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)

    metric_summary = validate(test_loader, model)
    print(f"mAP_50: {metric_summary['map_50'] * 100:.3f}")
    print(f"mAP_50_95: {metric_summary['map'] * 100:.3f}")
