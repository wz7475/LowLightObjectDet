from typing import List, Tuple

from labels_storage import exdark2coco_exceptions


def generate_exdark2coco_mapping(
    coco_categories: List[str], exdark_categories: List[str]
) -> Tuple[dict, dict]:
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
