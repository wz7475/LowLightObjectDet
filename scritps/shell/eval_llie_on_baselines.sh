#!/bin/bash

models=("wrapperfasterrcnn" "wrapperretinanet" "wrapperfcos" "wrapperdefdetr")
datamodules=("datamodule" "gamma" "hvicid")

for model in "${models[@]}"; do
  for datamodule in "${datamodules[@]}"; do
    python exdark/modeling/eval.py \
      model="$model" \
      datamodule="$datamodule"
  done
done
