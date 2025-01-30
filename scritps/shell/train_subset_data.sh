#!/bin/bash

models=("fasterrcnn" "retinanet" "fcos" "def_detr")
freeze_options=("true" "false")
limit_samples=(5 10 50 75 100 200 3000)

for model in "${models[@]}"; do
  for freeze in "${freeze_options[@]}"; do
    for samples in "${limit_samples[@]}"; do
      python exdark/modeling/train.py \
        model="$model" \
        model.freeze_backbone="$freeze" \
        datamodule.limit_to_n_samples="$samples" \
        datamodule.batch_size=4
    done
  done
done