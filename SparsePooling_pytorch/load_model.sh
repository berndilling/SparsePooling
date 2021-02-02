#!/bin/sh
########################################################################################################################################
##   ATTENTION: architecture is written to log and needs to be adapted in SparsePoolingModel.py if old architectures are re-loaded!   ##
########################################################################################################################################


echo "Load Model into IPython"
python -m SparsePooling_pytorch.load --model_path ./logs/SparsePooling --model_num 300
