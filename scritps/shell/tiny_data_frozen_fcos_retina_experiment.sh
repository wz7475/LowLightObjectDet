#!/bin/bash

echo "" > results_tiny_data_model.txt

models=("fcos" "retinanet")
for model in "${models[@]}";
do
  for num_samples in 5 10 50 75 100 200 3000
  do
    echo "$model $num_samples frozen"
     output=$(python exdark/modeling/train.py \
      "datamodule.limit_to_n_samples=$num_samples" \
      "model=$model" \
      "model.freeze_backbone=true" \
      "datamodule.batch_size=4") ;
     echo "$num_samples: $output" >> results_tiny_data_model.txt ;

     echo "$model $num_samples not-frozen"
     output=$(python exdark/modeling/train.py \
      "datamodule.limit_to_n_samples=$num_samples" \
      "model=$model" \
      "model.freeze_backbone=false" \
      "datamodule.batch_size=4") ;
     echo "$num_samples: $output" >> results_tiny_data_model.txt ;
  done
done
