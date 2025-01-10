import torch
from torch import tensor

from exdark.models.cocowrappers.detection_filter import filter_detections


def test_filter_detections_source_cat_not_present_in_target():
    outputs = [
        {'boxes': tensor([[264.9399, 68.7429, 609.7746, 314.7758],
                          [369.1588, 424.8215, 518.4522, 634.7663],
                          [310.8712, 295.9392, 400.9587, 385.9071]]),
         'labels': tensor([1, 2, 4]),
         'scores': tensor([0.7068, 0.6420, 0.4409])}
    ]
    categories_map = {1: 1, 2: 2}
    filtered = filter_detections(outputs, categories_map)
    assert len(filtered[0]["scores"]) == 2


def test_filter_detections_map_categories():
    detections = [
        {'boxes': tensor([[264.9399, 68.7429, 609.7746, 314.7758]]),
         'labels': tensor([1]),
         'scores': tensor([0.709])}
    ]
    categories_map = {1: 2}
    filtered = filter_detections(detections, categories_map)
    assert len(filtered[0]["boxes"]) == 1
    assert len(filtered[0]["labels"]) == 1
    assert len(filtered[0]["scores"]) == 1
    assert filtered[0]["labels"][0].item() == 2



def test_filter_detections_empty_target_list():
    detections = [
        {
            "boxes": [torch.tensor([9, 10, 11, 12])],
            "labels": [torch.tensor(3)],
            "scores": [0.7],
        }
    ]
    categories_map = {1: 1, 2: 2}
    filtered = filter_detections(detections, categories_map)
    assert len(filtered[0]["scores"]) == 0
