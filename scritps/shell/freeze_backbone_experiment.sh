#!/bin/bash

models=("fasterrcnn" "retinanet" "fcos")


echo "frozen - mAP" > results_freeze.txt
for i in "${models[@]}";
do
   output=$(python exdark/modeling/train.py \
    "model=$i" \
    "model.freeze_backbone=true") ;
   echo "$i: $output" >> results_freeze.txt ;
done

echo "not-frozen - mAP" >> results_freeze.txt
for i in "${models[@]}";
do
   output=$(python exdark/modeling/train.py \
    "model=$i" \
    "model.freeze_backbone=false") ;
   echo "$i: $output" >> results_freeze.txt ;
done
