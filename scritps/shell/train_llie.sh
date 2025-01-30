#!/bin/bash

models=("wrapperfasterrcnn" "wrapperretinanet" "wrapperfcos")
datamodules=("datamodule" "hvicid" "gamma")

for model in "${models[@]}"; do
  for datamodule in "${datamodules[@]}"; do
    python exdark/modeling/train.py \
      model="$model" \
      datamodule="$datamodule"
  done
done