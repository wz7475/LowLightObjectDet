#!/usr/bin/zsh

# expects imageclasslist.txt, directories ExDark and ExDark_Annno (see readme.md)

# make directory for data
mkdir ./dataset -p;

# create imageclasslist with class indexes translated to coco categories from torchvision
python3 parse_imgclasslist_classes_idx.py imageclasslist.txt dataset/imageclasslist.txt.coco  && echo "imageclasslist to coco"

# lower case for extensions and filenames (label and images are consistent in terms of case)
python3 names_tolower.py ./ExDark
python3 names_tolower.py ./ExDark_Annno

# create flattened version
mkdir ./dataset/flat -p;
python3 move_imgs_to_flat_dir.py ./ExDark ./dataset/flat && echo "flattened imgs";
mkdir ./dataset/flat_anno -p;
python3 move_imgs_to_flat_dir.py ./ExDark_Annno ./dataset/flat_anno && echo "flattened anno";

# rename labels in annotation files from ExDark names to COCO names
mkdir ./dataset/flat_anno_coco ;
python3 parse_anno_files_classes.py ./dataset/flat_anno ./dataset/flat_anno_coco && echo "anno to coco";


# train test val
python3 train_test_val.py dataset/imageclasslist.txt.coco ./dataset/split ./dataset/flat ./dataset/flat_anno_coco , && echo "train test val";
