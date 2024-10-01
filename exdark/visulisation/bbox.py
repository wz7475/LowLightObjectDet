import cv2
import numpy as np

from exdark.config import CLASSES_COCO, NUM_CLASSES_EXDARK

COLORS = [[0, 0, 0], [255, 0, 0]]
COLORS.append([255, 255, 0])  # Yellow
COLORS.append([0, 255, 0])  # Green
COLORS.append([0, 255, 255])  # Cyan
COLORS.append([255, 0, 255])  # Magenta
COLORS.append([128, 0, 0])  # Maroon
COLORS.append([128, 128, 0])  # Olive
COLORS.append([0, 128, 0])  # Dark Green
COLORS.append([128, 0, 128])  # Purple
COLORS.append([0, 128, 128])  # Teal
COLORS.append([192, 192, 192])  # Silver
COLORS.append([128, 128, 128])  # Gray


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


def draw_bbox_from_preds(image_rgb: np.array, outputs: dict, threshold: float = 0.5):
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = [CLASSES_COCO[i] for i in outputs[0]['labels'].cpu().numpy()]
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[CLASSES_COCO.index(class_name) % NUM_CLASSES_EXDARK]
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            cv2.rectangle(image_rgb,
                          (xmin, ymin),
                          (xmax, ymax),
                          color,
                          3)
            cv2.putText(image_rgb,
                        class_name,
                        (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        lineType=cv2.LINE_AA)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('Prediction', image_bgr)
    cv2.waitKey(0)


def draw_bbox_from_targets(image_rgb: np.array, target: dict):
    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    except:
        image_bgr = cv2.cvtColor(image_rgb.detach().cpu().numpy(), cv2.COLOR_RGB2BGR)

    for box_num in range(len(target['boxes'])):
        box = target['boxes'][box_num]
        label = CLASSES_COCO[target['labels'][box_num]]

        cv2.rectangle(
            image_bgr,
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 0, 255),
            1
        )

        draw_text(image_bgr, label, pos=(int(box[0]), int(box[1] - 5)))
    cv2.imshow('Image', image_bgr)
    cv2.waitKey(0)
