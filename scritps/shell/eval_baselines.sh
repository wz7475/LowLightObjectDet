#!/bin/bash

models=("wrapperfasterrcnn" "wrapperretinanet" "wrapperfcos" "wrapperdefdetr")

for model in "${models[@]}"; do
  python exdark/modeling/eval.py \
    model="$model"
done
