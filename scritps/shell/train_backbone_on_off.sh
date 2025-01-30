#!/bin/bash

models=("fasterrcnn" "retinanet" "fcos" "def_detr")
freeze_options=("false" "true")

for model in "${models[@]}"; do
  for freeze in "${freeze_options[@]}"; do
    python exdark/modeling/train.py \
      model="$model" \
      model.freeze_backbone="$freeze"
  done
done