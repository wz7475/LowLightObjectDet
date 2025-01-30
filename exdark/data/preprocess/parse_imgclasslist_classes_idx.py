"""
This script is used to parse the image_classlist.txt file and map the classes to the corresponding indexes.
"""

import os

import pandas as pd
from exdark.data.preprocess.labels_mappers import generate_exdark2coco_mapping
from exdark.data.preprocess.labels_storage import coco_labels, exdark_custon_labels


def parse_image_class_list_txt(
    input_file_path: str, output_path: str, labels_map: dict
):
    with open(input_file_path) as input_fp:
        lines = input_fp.readlines()
        lines[0] = lines[0].replace(" |", "")
        temp_path = f"{input_file_path}.temp"
        with open(temp_path, "w") as out_fp:
            out_fp.writelines(lines)
    anno_df = pd.read_csv(temp_path, delimiter=" ")
    anno_df["Class"] = anno_df["Class"].map(lambda label: labels_map[label])
    anno_df.to_csv(output_path, index=False)
    os.remove(temp_path)


if __name__ == "__main__":
    import sys

    class_list_path_in = sys.argv[1]
    class_list_path_out = sys.argv[2]

    labels_idx_map, _ = generate_exdark2coco_mapping(coco_labels, exdark_custon_labels)
    parse_image_class_list_txt(class_list_path_in, class_list_path_out, labels_idx_map)
