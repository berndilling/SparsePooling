#!/bin/sh

# Protocol for learning MaxPool by SFA within CNN
# assuming:

# architecture = [('BP', 100, 3, None, None), ('MaxPool', 100, 2, None, None), 
#                 ('BP', 200, 3, None, None), ('MaxPool', 200, 2, None, None), 
#                 ('BP', 400, 3, None, None), ('MaxPool', 400, 2, None, None)]
# and:
# architecture = [('BP', 100, 3, None, None), ('SFA', 100, 2, 0.5, 8), 
#                 ('BP', 200, 3, None, None), ('SFA', 200, 2, 0.5, 8), 
#                 ('BP', 400, 3, None, None), ('SFA', 400, 2, 0.5, 8)]

# to be changed in SparsePoolingModel.py

##############################################################################################


# 1. Train and test on classification task --end_to_end_supervised with downstream_classification. linear classifier will be added automatically and recurrence will be ignored
echo "Create untrained BP - MaxPool stack and train end-to-end and test on classification task"
python -m SparsePooling_pytorch.downstream_classification --save_dir SparsePooling_MaxPool_through_SFA --model_path ./logs/SparsePooling_MaxPool_through_SFA --num_epochs 100 --end_to_end_supervised --in_channels 400

!!! TODO add reloading of learned BP weights into BP/SFA stack !!!
# 2. Retrain with SFA on moving dataset.
python -m SparsePooling_pytorch.main --save_dir SparsePooling_MaxPool_through_SFA --num_epochs 100 --dataset_type "moving" --dataset "stl10" --patch_size 25 --tau 0.01 --sequence_length 20

!!! TODO: shut down learning in "SC"/"BP" layers !!!

# 3. Test model on classification task

# 4. Potentially retrain and test linear downstream classifier with downstream_classification

# 5. Investigate learned MaxPooling layers