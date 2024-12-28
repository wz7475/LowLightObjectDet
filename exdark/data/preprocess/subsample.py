"""
given dataset wih paired images and annotations create subset of given size
"""

import os
import random
import shutil
import argparse


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
    os.makedirs(destination_dir, exist_ok=True)
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


def main():
    parser = argparse.ArgumentParser(description="Sample images and their annotations from a directory.")
    parser.add_argument("dir_for_sampling", type=str, help="The directory containing the images to sample from.")
    parser.add_argument("destination_dir", type=str, help="The directory where the sampled images and annotations will be copied to.")
    parser.add_argument("num_samples", type=int, help="The number of images (and their annotations) to sample and copy.")
    
    args = parser.parse_args()
    
    try:
        create_subset_dir(args.dir_for_sampling, args.destination_dir, args.num_samples)
    except NonEnoughImagesError:
        print("Not enough images available in the source directory to sample the requested number of images.")

if __name__ == "__main__":
    main()