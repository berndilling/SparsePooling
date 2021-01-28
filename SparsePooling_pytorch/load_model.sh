#!/bin/sh

echo "Load Model into IPython"
python -m SparsePooling_pytorch.load --model_path ./logs/SparsePooling --model_num 300
