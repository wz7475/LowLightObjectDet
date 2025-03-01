import os
from argparse import ArgumentParser
import albumentations as A
import cv2

SIZE = (640, 640)
transform = A.RandomGamma((50, 51), p=1)

def process_images(input_path: str, output_path: str) -> None:
    img = cv2.imread(input_path)
    img = cv2.resize(img, SIZE)
    # augmented_img = transform(image=img)["image"]
    augmented_img = img
    cv2.imwrite(output_path, augmented_img)

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output

    img_paths = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.split(".")[-1].lower().lower() in ["jpg", "jpeg", "png", "ppm"]
    ]
    total = len(img_paths)
    os.makedirs(output_dir, exist_ok=True)
    for idx, img_path in enumerate(img_paths):
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        process_images(img_path, output_path)
        print(f"{idx}/{total}")

if __name__ == "__main__":
    main()