import cv2
import numpy as np

from exdark.config import CLASSES_COCO


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=1,
              font_thickness=1,
              text_color=(0, 0, 255),
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def draw_bbox(image_rgb: np.array, target: dict, score: float = None):
    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    except:
        image_bgr = cv2.cvtColor(image_rgb.detach().cpu().numpy(), cv2.COLOR_RGB2BGR)

    for box_num in range(len(target['boxes'])):
        box = target['boxes'][box_num]
        label = CLASSES_COCO[target['labels'][box_num]]
        text_for_bbox = label
        if score:
            text_for_bbox = f"{label} - {int(score * 100)}%"

        cv2.rectangle(
            image_bgr,
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 0, 255),
            1
        )

        draw_text(image_bgr, text_for_bbox, pos=(int(box[0]), int(box[1] - 5)))
    cv2.imshow('Image', image_bgr)
    cv2.waitKey(0)
