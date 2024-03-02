from typing import List, Iterable

import torch


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


def convert_coco2_exdark(labels: torch.Tensor, mapping: dict) -> list:
    converted_labels = []
    for src_label in labels:
        converted_labels.append(mapping[src_label.item()])
    return converted_labels



