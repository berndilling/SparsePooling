#!/bin/sh

# TODO finish this
echo "Testing the SparsePooling Model on downstream image classification"
python -m SparsePooling_pytorch.downstream_classification --model_path ./logs/testpool_white --end_to_end_supervised