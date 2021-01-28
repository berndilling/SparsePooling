#!/bin/sh

echo "Training the SparsePooling Model"
python -m SparsePooling_pytorch.main --save_dir testpool_white --num_epochs 50 --dataset_type "moving" --preprocess "whiten"

echo "Testing the SparsePooling Model on downstream image classification"
python -m SparsePooling_pytorch.downstream_classification --model_path ./logs/testpool_white --model_num 300  --download_dataset 