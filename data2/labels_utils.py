import csv
import os
import shutil
from typing import List, Iterable, Tuple
import pandas as pd
import torch

from labels_storage import coco_labels, exdark_labels


def generate_coco2exdark_mapping(coco_categories: List[str], exdark_categories: List[str]):
    convertion_excpetions = {
        "motorcycle": "motorbike",
        "person": "people",
        "dining table": "table"
    }

    mapping_idx = {}
    mapping_cat = {}
    for idx_coco, category_coco in enumerate(coco_categories):
        if category_coco in convertion_excpetions:
            category_coco = convertion_excpetions[category_coco]
        try:
            idx_exdark = exdark_categories.index(category_coco)
        except ValueError:
            idx_exdark = exdark_categories.index("other")
        mapping_idx[idx_coco] = idx_exdark
        mapping_cat[category_coco] = exdark_categories[idx_exdark]
    return mapping_idx, mapping_cat


def generate_exdark2coco_mapping(coco_categories: List[str], exdark_categories: List[str]) -> Tuple[dict, dict]:
    convertion_excpetions = {
        "motorbike": "motorcycle",
        "people": "person",
        "table": "dining table"
    }

    mapping_idx = {}
    mapping_cat = {}
    for idx_exdark, category_exdark in enumerate(exdark_categories):
        if category_exdark in convertion_excpetions:
            category_exdark_changed = convertion_excpetions[category_exdark]
            idx_coco = coco_categories.index(category_exdark_changed)
        else:
            idx_coco = coco_categories.index(category_exdark)
        mapping_idx[idx_exdark] = idx_coco
        mapping_cat[category_exdark] = coco_categories[idx_coco]
    return mapping_idx, mapping_cat


def parse_image_class_list_txt(input_file_path: str, output_path: str, labels_map: dict):
    with open(input_file_path) as input_fp:
        lines = input_fp.readlines()
        lines[0] = lines[0].replace(" |", "")
        temp_path = f"{input_file_path}.temp"
        with open(temp_path, "w") as out_fp:
            out_fp.writelines(lines)
    anno_df = pd.read_csv(temp_path, delimiter=" ")
    anno_df['Class'] = anno_df['Class'].map(lambda label: labels_map[label])
    input_file_name = os.path.basename(input_file_path)
    anno_df.to_csv(output_path, index=False)
    os.remove(temp_path)

def parse_exdark_anno_dir_flat(input_dir: str, output_dir: str, labels_map: dict):
    anno_file_columns = ["class"] + ["_"] * 11
    for anno_file in os.listdir(input_dir):
        anno_file_path = os.path.join(input_dir, anno_file)
        anno_df = pd.read_csv(anno_file_path, header=None, names=anno_file_columns)
        anno_df["class"] = anno_df["class"].map(lambda class_name: labels_map[class_name])
        output_file_path = os.path.join(output_dir, anno_file)
        anno_df.to_csv(output_file_path, header=False)


if __name__ == "__main__":
    import sys
    class_list_path = sys.argv[1]
    output_path = sys.argv[2]

    labels_idx_map, labels_names_map = generate_exdark2coco_mapping(coco_labels, exdark_labels)
    parse_image_class_list_txt(class_list_path, output_path, labels_idx_map)
