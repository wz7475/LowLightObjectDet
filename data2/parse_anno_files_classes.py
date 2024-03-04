import os

import pandas as pd
from labels_mappers import generate_exdark2coco_mapping
from labels_storage import coco_labels, exdark_labels


def parse_exdark_anno_dir_flat(input_dir: str, output_dir: str, labels_map: dict):
    anno_file_columns = ["class"] + [str(x) for x in range(11)]
    for anno_file in os.listdir(input_dir):
        anno_file_path = os.path.join(input_dir, anno_file)
        anno_df = pd.read_csv(anno_file_path, skiprows=[0], delimiter=" ", header=None, names=anno_file_columns)
        anno_df["class"] = anno_df["class"].map(lambda class_name: labels_map[class_name.lower()])
        output_file_path = os.path.join(output_dir, anno_file)
        anno_df.to_csv(output_file_path, header=False, index=False)


if __name__ == "__main__":
    import sys

    anno_dir_in = sys.argv[1]
    anno_dir_out = sys.argv[2]

    _, labels_names_map = generate_exdark2coco_mapping(coco_labels, exdark_labels)
    parse_exdark_anno_dir_flat(anno_dir_in, anno_dir_out, labels_names_map)
