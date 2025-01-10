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


PREDEFINED_PREDICTIONS = [
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


@pytest.fixture
def mock_torchvision_model():
    model = MagicMock()
    model.return_value = PREDEFINED_PREDICTIONS
    return model


@pytest.fixture
def mock_transformers_wrapper():
    wrapper = COCOWrapperTransformers()
    wrapper.categories_map = {1: 1, 2: 2, 3: 3, 4: 4}
    wrapper.image_processor = MagicMock()
    wrapper.image_processor.post_process_object_detection.return_value = (
        PREDEFINED_PREDICTIONS
    )
    wrapper.image_processor.return_value = {"pixel_values": torch.rand(1, 3, 224, 224)}
    return wrapper


@pytest.fixture
def mock_lightning_torchvsionwrapper(mock_torchvision_model):
    wrapper = COCOWrapperTorchvision(mock_torchvision_model)
    wrapper.categories_filter = {1: 1, 2: 2, 3: 3, 4: 4}
    return wrapper


def test_50_map(mock_dataloader, mock_lightning_torchvsionwrapper):
    images, targets = mock_dataloader
    mock_lightning_torchvsionwrapper.test_step((images, targets), 0)
    assert mock_lightning_torchvsionwrapper.metric.compute()["map_50"] == 1


def test_transformers_50_map(mock_dataloader, mock_transformers_wrapper):
    images, targets = mock_dataloader
    mock_transformers_wrapper.test_step((images, targets), 0)
    assert mock_transformers_wrapper.metric.compute()["map_50"] == 1
