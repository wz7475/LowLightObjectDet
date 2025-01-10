import torch

from exdark.models.cocowrappers.detection_filter import filter_detections


def test_filter_detections_source_cat_not_present_in_target():
    detections = [
        {
            "boxes": [torch.tensor([1, 2, 3, 4])],
            "labels": [torch.tensor(1)],
            "scores": [0.9],
        },
        {
            "boxes": [torch.tensor([5, 6, 7, 8])],
            "labels": [torch.tensor(2)],
            "scores": [0.8],
        },
        {
            "boxes": [torch.tensor([9, 10, 11, 12])],
            "labels": [torch.tensor(3)],
            "scores": [0.7],
        },
    ]
    categories_map = {1: 1, 2: 2}
    filtered = filter_detections(detections, categories_map)
    assert len(filtered) == 2


def test_filter_detections_map_categories():
    detections = [
        {
            "boxes": [torch.tensor([1, 2, 3, 4])],
            "labels": [torch.tensor(1)],
            "scores": [0.9],
        }
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
    assert len(filtered) == 0
