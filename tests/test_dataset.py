from unittest.mock import patch

import pytest

from exdark.data.datasets import ExDarkDataset


@pytest.fixture
def sample_dataset():
    return ExDarkDataset(dir_path="/fake/path", width=640, height=480)


@pytest.fixture
def mock_image_files():
    return ["/fake/path/img1.jpg", "/fake/path/img2.jpg", "/fake/path/img3.png"]


@patch("glob.glob")
def test_init_with_limit_samples(mock_glob, mock_image_files):
    mock_glob.return_value = mock_image_files
    dataset = ExDarkDataset(
        dir_path="/test/path", width=640, height=480, limit_to_n_samples=2
    )
    assert len(dataset.all_images) == 2


def test_basic__get_bbox(sample_dataset):
    raw_bbox = [100, 100, 200, 150]
    img_width = 1000
    img_height = 800

    bbox = sample_dataset._get_bbox(raw_bbox, img_width, img_height)

    assert len(bbox) == 4
    assert bbox[0] >= 0
    assert bbox[1] >= 0
    assert bbox[2] <= sample_dataset.width
    assert bbox[3] <= sample_dataset.height


def test_get_bbox_boundary_case(sample_dataset):
    # outside image bounds
    raw_bbox = [-100, -100, 2000, 2000]
    img_width = 1000
    img_height = 800

    bbox = sample_dataset._get_bbox(raw_bbox, img_width, img_height)

    assert bbox[0] == 0
    assert bbox[1] == 0
    assert bbox[2] == sample_dataset.width
    assert bbox[3] == sample_dataset.height
