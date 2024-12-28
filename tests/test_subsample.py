import os
import shutil
import tempfile

import pytest

from exdark.data.preprocess.subsample import create_subset_dir, NonEnoughImagesError


@pytest.fixture
def setup_directories():
    source_dir = tempfile.mkdtemp()
    destination_dir = tempfile.mkdtemp()

    for i in range(5):
        with open(os.path.join(source_dir, f"image_{i}.jpg"), 'w') as f:
            f.write("image content")
        with open(os.path.join(source_dir, f"image_{i}.jpg.txt"), 'w') as f:
            f.write("annotation content")

    yield source_dir, destination_dir

    shutil.rmtree(source_dir)
    shutil.rmtree(destination_dir)

def test_create_subset_dir_not_enough_images(setup_directories):
    source_dir, destination_dir = setup_directories
    with pytest.raises(NonEnoughImagesError):
        create_subset_dir(source_dir, destination_dir, 10)

def test_create_subset_dir_success(setup_directories):
    source_dir, destination_dir = setup_directories
    create_subset_dir(source_dir, destination_dir, 3)

    copied_images = [f for f in os.listdir(destination_dir) if f.endswith(".jpg")]
    copied_annotations = [f for f in os.listdir(destination_dir) if f.endswith(".jpg.txt")]
    
    assert len(copied_images) == 3
    assert len(copied_annotations) == 3
    for img in copied_images:
        assert os.path.exists(os.path.join(destination_dir, img))
        assert os.path.exists(os.path.join(destination_dir, f"{img}.txt"))