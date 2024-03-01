import os
import sys
from enlighten_inference import EnlightenOnnxModel
import cv2
from matplotlib import pyplot as plt


dir_path = sys.argv[1]
dest_path = sys.argv[2]

model = EnlightenOnnxModel()
img_exts = ['.jpeg', '.jpg', '.JPG', '.JPEG', '.png']

for file in os.listdir(dir_path):
    ext = os.path.splitext(file)[1]
    if ext in img_exts:
        print(file)
        img_path = os.path.join(dir_path, file)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        img = model.predict(img)

        dest_path = os.path.join(dest_path, file)
        cv2.imwrite(dest_path, img)