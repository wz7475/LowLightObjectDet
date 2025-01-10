import cv2
import numpy as np

from exdark.data.config import CLASSES_COCO, NUM_CLASSES_EXDARK
from exdark.data.preprocess.labels_storage import (
    exdark_idx2label,
    exdark_coco_like_labels,
)

COLORS = [[0, 0, 0], [0, 0, 255]]
COLORS.append([255, 255, 0])
COLORS.append([0, 255, 0])
COLORS.append([0, 255, 255])
COLORS.append([255, 0, 255])
COLORS.append([128, 0, 0])
COLORS.append([128, 128, 0])
COLORS.append([255, 128, 0])
COLORS.append([128, 0, 128])
COLORS.append([0, 128, 128])
COLORS.append([192, 192, 192])
COLORS.append([128, 128, 128])


def draw_text(
    img,
    text,
    font=cv2.FONT_HERSHEY_PLAIN,
    pos=(0, 0),
    font_scale=1,
    font_thickness=1,
    text_color=(0, 0, 255),
    text_color_bg=(0, 0, 0),
):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(
        img,
        text,
        (x, y + text_h + font_scale - 1),
        font,
        font_scale,
        text_color,
        font_thickness,
    )

    return text_size


def draw_bbox(img, bbox_pascal_voc, color=(0, 0, 255)):
    cv2.rectangle(
        img,
        (int(bbox_pascal_voc[0]), int(bbox_pascal_voc[1])),
        (int(bbox_pascal_voc[2]), int(bbox_pascal_voc[3])),
        color,
        1,
    )


def draw_bbox_with_text(img, bbox_pascal_voc, text, color=(0, 0, 255)):
    draw_bbox(img, bbox_pascal_voc, color)
    draw_text(
        img,
        text,
        text_color=color,
        pos=(int(bbox_pascal_voc[0]), int(bbox_pascal_voc[1] - 5)),
    )


def preprocess_predictions(outputs, threshold):
    boxes, scores, pred_classes = [], [], []
    if len(outputs[0]["boxes"]) != 0:
        boxes = outputs[0]["boxes"].data.numpy()
        scores = outputs[0]["scores"].data.numpy()
        boxes = boxes[scores >= threshold].astype(
            np.int32
        )  # scores are sorted from highest to lowest
        pred_classes = outputs[0]["labels"].cpu().numpy()
    return boxes, scores, pred_classes


def print_predictions(boxes, scores, pred_classes):
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        class_name = exdark_coco_like_labels[pred_classes[idx]]
        text_to_write = f"{class_name} {int(score * 100)}%"
        print(text_to_write)


def draw_bbox_from_preds(image_rgb: np.array, boxes, scores, pred_classes):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for j, (box, score) in enumerate(zip(boxes, scores)):
        class_name = exdark_coco_like_labels[pred_classes[j]]
        text_to_write = f"{class_name} {int(score * 100)}%"
        color = COLORS[CLASSES_COCO.index(class_name) % NUM_CLASSES_EXDARK]
        draw_bbox_with_text(image_bgr, box, text_to_write, color)
    cv2.imshow("Prediction", image_bgr)
    cv2.waitKey(0)


def draw_bbox_from_targets(image_tensor, target: dict):
    image_rgb = (
        image_tensor.permute(1, 2, 0).cpu().numpy()
    )
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for box_num in range(len(target["boxes"])):
        box = target["boxes"][box_num]
        label_num = target["labels"][box_num].item()
        label_name = exdark_idx2label[label_num]

        color = COLORS[label_num]
        draw_bbox_with_text(image_bgr, box, label_name, color)
    cv2.imshow("Image", image_bgr)
    cv2.waitKey(0)
