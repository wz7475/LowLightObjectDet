"""
This script is used to split the ExDark dataset into train, val, and test sets.
"""

import os
import shutil
import sys

annotations_info_path = sys.argv[1]
dest_dir = sys.argv[2]
img_dir = sys.argv[3]
annotation_dir = sys.argv[4]
delimiter = sys.argv[5]


with open(annotations_info_path, "r") as f:
    annotations_info = f.readlines()
annotations_info = [x.strip() for x in annotations_info[1:]]

train_val_test_map = {"1": "train", "2": "val", "3": "test"}
for dir in train_val_test_map.values():
    os.makedirs(os.path.join(dest_dir, dir), exist_ok=True)


for annotation_info in annotations_info:
    annotation_info = annotation_info.split(delimiter)
    annotation_path = annotation_info[0]
    train_test_val_idx = annotation_info[4]
    train_test_val = train_val_test_map[train_test_val_idx]

    img_name = os.path.basename(annotation_path).lower()
    img_src_path = os.path.join(img_dir, img_name)
    annotaion_name = img_name + ".txt"
    annotation_src_path = os.path.join(annotation_dir, annotaion_name)

    dest_img_path = os.path.join(
        dest_dir, train_test_val, os.path.basename(annotation_path)
    )
    dest_annotaion_path = os.path.join(dest_dir, train_test_val, annotaion_name)

    shutil.copy(img_src_path, dest_img_path)
    shutil.copy(annotation_src_path, dest_annotaion_path)
