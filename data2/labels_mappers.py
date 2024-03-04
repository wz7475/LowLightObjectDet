from typing import List, Tuple
from labels_storage import coco2exdark_exceptions, exdark2coco_exceptions


def generate_coco2exdark_mapping(coco_categories: List[str], exdark_categories: List[str]):
    mapping_idx = {}
    mapping_cat = {}
    for idx_coco, category_coco in enumerate(coco_categories):
        if category_coco in coco2exdark_exceptions:
            category_coco = coco2exdark_exceptions[category_coco]
        try:
            idx_exdark = exdark_categories.index(category_coco)
        except ValueError:
            idx_exdark = exdark_categories.index("other")
        mapping_idx[idx_coco] = idx_exdark
        mapping_cat[category_coco] = exdark_categories[idx_exdark]
    return mapping_idx, mapping_cat


def generate_exdark2coco_mapping(coco_categories: List[str], exdark_categories: List[str]) -> Tuple[dict, dict]:
    mapping_idx = {}
    mapping_cat = {}
    for idx_exdark, category_exdark in enumerate(exdark_categories):
        if category_exdark in exdark2coco_exceptions:
            category_exdark_changed = exdark2coco_exceptions[category_exdark]
            idx_coco = coco_categories.index(category_exdark_changed)
        else:
            idx_coco = coco_categories.index(category_exdark)
        mapping_idx[idx_exdark] = idx_coco
        mapping_cat[category_exdark] = coco_categories[idx_coco]
    return mapping_idx, mapping_cat
