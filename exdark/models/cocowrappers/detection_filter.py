import torch

def filter_detections(detections: list[dict], categories_map: dict) -> list[dict]:
    filtered_detections_list = []
    for detections_dict in detections:
        filtered_detections_dict = {"boxes": [], "labels": [], "scores": []}
        for box, label_tensor, score in zip(
            detections_dict["boxes"],
            detections_dict["labels"],
            detections_dict["scores"],
        ):
            label = label_tensor.item()
            if label not in categories_map:
                continue
            label = categories_map[label]
            filtered_detections_dict["boxes"].append(box.tolist())
            filtered_detections_dict["labels"].append(label)
            filtered_detections_dict["scores"].append(score)
            filtered_detections_list.append(
                dict((k, torch.tensor(v)) for k, v in filtered_detections_dict.items())
            )
    return filtered_detections_list
