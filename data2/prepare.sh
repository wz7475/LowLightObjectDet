#!/usr/bin/zsh

# expects imageclasslist.txt, directories ExDark and ExDark_Annno (see readme.md)

# make directory for data
mkdir ./dataset -p;

# create imageclasslist with class translated to coco categories from torchvision
python3 labels_utils.py imageclasslist.txt dataset/imageclasslist.txt.coco && echo "to coco"

# lower case for extensions and filenames (label and images are consistent in terms of case)
python3 names_tolower.py ./ExDark
python3 names_tolower.py ./ExDark_Annno

# create flattened version
mkdir ./dataset/flat -p;
python3 move_imgs_to_flat_dir.py ./ExDark ./dataset/flat && echo "flattened imgs";
mkdir ./dataset/flat_anno -p;
python3 move_imgs_to_flat_dir.py ./ExDark_Annno ./dataset/flat_anno && echo "flattened anno";


# train test val
python3 train_test_val.py dataset/imageclasslist.txt.coco ./dataset/split ./dataset/flat ./dataset/flat_anno , && echo "train test val"
