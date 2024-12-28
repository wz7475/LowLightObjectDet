"""
given dataset wih paired images and annotations create subset of given size
"""

import os
import random
import shutil


class NonEnoughImagesError(Exception):
    pass


def get_annotation_path(img_path: str) -> str:
    return f"{img_path.lower()}.txt"


def create_subset_dir(dir_for_sampling: str, destination_dir: str, num_samples: int) -> None:
    """
    Creates a subset directory by sampling a specified number of images and their corresponding annotations 
    from the source directory and copying them to the destination directory.

    Args:
        dir_for_sampling (str): The directory containing the images to sample from.
        destination_dir (str): The directory where the sampled images and annotations will be copied to.
        num_samples (int): The number of images (and their annotations) to sample and copy.

    Raises:
        NonEnoughImagesError: If the number of available images in the source directory is less than the number of samples requested.

    Returns:
        None
    """
    img_filenames = [
        filename for filename in os.listdir(dir_for_sampling) if not filename.endswith("txt")
    ]
    if len(img_filenames) < num_samples:
        raise NonEnoughImagesError()
    for img_filename in random.sample(img_filenames, num_samples):
        annotation_filename = get_annotation_path(img_filename)
        src_img = os.path.join(dir_for_sampling, img_filename)
        dest_img = os.path.join(destination_dir, img_filename)
        src_annotation = os.path.join(dir_for_sampling, annotation_filename)
        dest_annotation = os.path.join(destination_dir, annotation_filename)
        shutil.copyfile(src_img, dest_img)
        shutil.copyfile(src_annotation, dest_annotation)
