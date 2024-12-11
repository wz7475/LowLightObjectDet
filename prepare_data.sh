#!/bin/bash

images_url="https://drive.usercontent.google.com/download?id=1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx&export=download&authuser=0&confirm=t&uuid=3232338f-26c7-4599-9f35-f5c93b8241e4&at=AO7h07dJS6_aNoNQUhl1VrZ_c3gt:1726826664489"

cd "./data" || exit ;

wget $images_url -O "ExDark.zip" ;
unzip "ExDark.zip" ;
unzip "ExDark_Annno.zip" ;
rm ExDark.zip ;


# make directory for data
mkdir -p ./dataset ;

# create imageclasslist with class indexes translated to coco categories from torchvision
python parse_imgclasslist_classes_idx.py imageclasslist.txt dataset/imageclasslist.txt.coco  && echo "imageclasslist to coco"

# lower case for extensions and filenames (label and images are consistent in terms of case)
python names_tolower.py ./ExDark
python names_tolower.py ./ExDark_Annno

# create flattened version
mkdir ./dataset/flat -p;
python move_imgs_to_flat_dir.py ./ExDark ./dataset/flat && echo "flattened imgs";
mkdir ./dataset/flat_anno -p;
python move_imgs_to_flat_dir.py ./ExDark_Annno ./dataset/flat_anno && echo "flattened anno";

rm -rf ./ExDark_Annno ExDark;

# rename labels in annotation files from ExDark names to COCO names
mkdir ./dataset/flat_anno_coco ;
python parse_anno_files_classes.py ./dataset/flat_anno ./dataset/flat_anno_coco && echo "anno to coco";

# train test val
python train_test_val.py dataset/imageclasslist.txt.coco ./dataset/split ./dataset/flat ./dataset/flat_anno_coco , && echo "train test val";

rm -rf ./dataset/flat ./dataset/flat ./dataset/flat_anno ./dataset/flat_anno_coco ;

cd ".." ;
