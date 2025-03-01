import os
import shutil
from argparse import ArgumentParser


def main():
    """take n dirs (per llie) as input, each of them have to contain files with same names
    ->produce dir of dir per image"""
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs="+", help="list of input directories", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    input = args.input
    output_dir = args.output

    first_input_dir = input[0]
    img_filenames = [
        filename
        for filename in os.listdir(first_input_dir)
        if filename.split(".")[-1].lower().lower() in ["jpg", "jpeg", "png", "ppm"]
    ]
    total = len(img_filenames)
    for idx, img_filename in enumerate(img_filenames):
        filename_no_ext, extention = os.path.splitext(img_filename)
        dir_for_img = os.path.join(output_dir, filename_no_ext)
        os.makedirs(dir_for_img, exist_ok=True)
        for llie_dir_path, llie_name in zip(input[0::2], input[1::2]):
            input_path = os.path.join(llie_dir_path, img_filename)
            dest_path = os.path.join(dir_for_img, f"{llie_name}{extention}")
            shutil.copy(input_path, dest_path)
        print(f"{idx}/{total}")

if __name__ == "__main__":
    main()