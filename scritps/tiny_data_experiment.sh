#!/bin/bash

echo "num_samples: - mAP" > results_tiny_data.txt

for i in 5 10 50 75 100 200 3000
do
   output=$(python exdark/modeling/train.py \
    "datamodule.limit_to_n_samples=$i" \
    "datamodule.batch_size=4") ;
   echo "$i: $output" >> results_tiny_data.txt ;
done
#   output=$(python exdark/modeling/train.py \
#    "datamodule.limit_to_n_samples=$i" \
#    "datamodule.batch_size=4" \
#    "trainer.max_epochs=5") ;