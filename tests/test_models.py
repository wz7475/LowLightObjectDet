from unittest.mock import MagicMock

import pytest
import torch

from exdark.models.cocowrappers.cocowrappertorchvision import COCOWrapperTorchvision
from exdark.models.cocowrappers.cocowrappertransformers import COCOWrapperTransformers


@pytest.fixture
def mock_dataloader():
    images = [torch.rand(3, 224, 224) for _ in range(4)]
    targets = [
        {"boxes": torch.tensor([[10, 10, 50, 50]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[20, 20, 60, 60]]), "labels": torch.tensor([2])},
        {"boxes": torch.tensor([[30, 30, 70, 70]]), "labels": torch.tensor([3])},
        {"boxes": torch.tensor([[40, 40, 80, 80]]), "labels": torch.tensor([4])},
    ]
    return images, targets


CATEGORIES_MAP = {1: 1, 2: 2, 3: 3, 4: 4}

PREDEFINED_PREDICTIONS_1_MAP = [
    {
        "boxes": torch.tensor([[10, 10, 50, 50]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    },
    {
        "boxes": torch.tensor([[20, 20, 60, 60]]),
        "labels": torch.tensor([2]),
        "scores": torch.tensor([0.8]),
    },
    {
        "boxes": torch.tensor([[30, 30, 70, 70]]),
        "labels": torch.tensor([3]),
        "scores": torch.tensor([0.7]),
    },
    {
        "boxes": torch.tensor([[40, 40, 80, 80]]),
        "labels": torch.tensor([4]),
        "scores": torch.tensor([0.6]),
    },
]

PREDEFINED_PREDICTIONS_1_MAP50 = [
    {
        "boxes": torch.tensor([[10, 10, 50, 31]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    },
    {
        "boxes": torch.tensor([[20, 20, 60, 41]]),
        "labels": torch.tensor([2]),
        "scores": torch.tensor([0.8]),
    },
    {
        "boxes": torch.tensor([[30, 30, 70, 51]]),
        "labels": torch.tensor([3]),
        "scores": torch.tensor([0.7]),
    },
    {
        "boxes": torch.tensor([[40, 40, 80, 61]]),
        "labels": torch.tensor([4]),
        "scores": torch.tensor([0.6]),
    },
]


@pytest.fixture
def mock_transformerswrapper(request):
    predictions, categories = request.param
    wrapper = COCOWrapperTransformers()
    wrapper._get_categories_map = lambda: categories
    wrapper.image_processor = MagicMock()
    wrapper.image_processor.post_process_object_detection.return_value = predictions
    wrapper.image_processor.return_value = {"pixel_values": torch.rand(1, 3, 224, 224)}
    return wrapper


@pytest.fixture
def mock_torchvisionwrapper(request):
    predictions, categories = request.param
    base_mock_torchvision_model = MagicMock()
    base_mock_torchvision_model.return_value = predictions
    wrapper = COCOWrapperTorchvision(base_mock_torchvision_model)
    wrapper.categories_filter = categories
    return wrapper


@pytest.mark.parametrize(
    "mock_torchvisionwrapper,expected_map,expected_map50",
    [
        ((PREDEFINED_PREDICTIONS_1_MAP, CATEGORIES_MAP), 1, 1),
        ((PREDEFINED_PREDICTIONS_1_MAP50, CATEGORIES_MAP), 0.1, 1),
    ],
    indirect=["mock_torchvisionwrapper"],
)
def test_torchvision_metrics(
    mock_dataloader, mock_torchvisionwrapper, expected_map, expected_map50
):
    images, targets = mock_dataloader
    mock_torchvisionwrapper.test_step((images, targets), 0)
    metrics = mock_torchvisionwrapper.metric.compute()
    assert metrics["map_50"] == expected_map50
    assert metrics["map"] == expected_map


@pytest.mark.parametrize(
    "mock_transformerswrapper,expected_map,expected_map50",
    [
        ((PREDEFINED_PREDICTIONS_1_MAP, CATEGORIES_MAP), 1, 1),
        ((PREDEFINED_PREDICTIONS_1_MAP50, CATEGORIES_MAP), 0.1, 1),
    ],
    indirect=["mock_transformerswrapper"],
)
def test_transformers_50_map(
    mock_dataloader, mock_transformerswrapper, expected_map, expected_map50
):
    images, targets = mock_dataloader
    mock_transformerswrapper.test_step((images, targets), 0)
    assert mock_transformerswrapper.metric.compute()["map_50"] == expected_map50
    assert mock_transformerswrapper.metric.compute()["map"] == expected_map
