#!/bin/sh

echo "Training the SparsePooling Model"
python -m SparsePooling_pytorch.main --save_dir test

echo "Testing the SparsePooling Model on downstream image classification"
python -m SparsePooling_pytorch.downstream_classification --model_path ./logs/test --model_num 299