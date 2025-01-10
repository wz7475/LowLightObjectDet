import pytest
import torch
from unittest.mock import patch, MagicMock

from torch import tensor

from exdark.modeling.visualizedata import visualize_data, TooManySamplesError


def generate_random_image_and_target():
    image = torch.rand(3, 640, 640)
    target = {
        "boxes": tensor([[218.0, 46.0, 402.0, 240.0], [248.0, 4.0, 400.0, 196.0]]),
        "labels": tensor([1, 11]),
        "size": tensor([640, 640]),
        "image_id": tensor([0]),
    }
    return (image,), (target,)


def test_visualize_data_too_many_samples():
    mock_datamodule = MagicMock()
    mock_datamodule.batch_size = 1
    mock_dataloader = [generate_random_image_and_target() for _ in range(2)]
    mock_datamodule.test_dataloader.return_value = mock_dataloader

    with pytest.raises(TooManySamplesError):
        visualize_data(mock_datamodule, 3)


def test_visualize_data_enough_samples():
    mock_datamodule = MagicMock()
    mock_datamodule.batch_size = 1
    mock_dataloader = [generate_random_image_and_target() for _ in range(2)]
    mock_datamodule.test_dataloader.return_value = mock_dataloader

    with patch("cv2.imshow"), patch("cv2.waitKey"):
        visualize_data(mock_datamodule, 2)
        assert mock_datamodule.test_dataloader.call_count == 1


def test_visualize_data_zero_samples():
    mock_datamodule = MagicMock()
    mock_datamodule.batch_size = 1
    mock_dataloader = [generate_random_image_and_target() for _ in range(2)]
    mock_datamodule.test_dataloader.return_value = mock_dataloader

    with patch("cv2.imshow"), patch("cv2.waitKey"):
        visualize_data(mock_datamodule, 0)
        assert mock_datamodule.test_dataloader.call_count == 1
