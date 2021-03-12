#!/bin/sh

# Protocol for learning MaxPool by SFA within CNN

##############################################################################################

# TODO implement automated switching between architectures!
# 1. Train and test on classification task --end_to_end_supervised with downstream_classification. linear classifier will be added automatically and recurrence will be ignored
# assuming (to be changed in SparsePoolingModel.py):
# architecture = [('BP', 100, 3, None, None), ('MaxPool', 100, 2, None, None), 
#                 ('BP', 200, 3, None, None), ('MaxPool', 200, 2, None, None), 
#                 ('BP', 400, 3, None, None), ('MaxPool', 400, 2, None, None)]
echo "Create untrained BP - MaxPool stack and train end-to-end and test on classification task"
python -m SparsePooling_pytorch.downstream_classification --save_dir SparsePooling_MaxPool_through_SFA_end_to_end_supervised --model_path ./logs/SparsePooling_MaxPool_through_SFA_end_to_end_supervised --num_epochs 100 --end_to_end_supervised --in_channels 400

echo "Load model"
python -m SparsePooling_pytorch.load --model_path ./logs/SparsePooling_MaxPool_through_SFA_end_to_end_supervised --model_num 99 --dataset_type "moving" --dataset "stl10" --patch_size 25
#  e.g. to measure sparsity
# >> out, _ = model(next(iter(train_loader)), up_to_layer=5)
# >> utils.getsparsity(out)

# 2. Retrain with SFA on moving dataset.
# assuming (to be changed in SparsePoolingModel.py):
# architecture = [('BP', 100, 3, None, None), ('SFA', 100, 2, 0.38, 8), 
#                 ('BP', 200, 3, None, None), ('SFA', 200, 2, 0.95, 8), 
#                 ('BP', 400, 3, None, None), ('SFA', 400, 2, 0.87, 8)]
# p -values were measured form BP-MaxPool network on STL-10 data
echo "Retrain SFA-layers on moving dataset"
python -m SparsePooling_pytorch.main --save_dir SparsePooling_MaxPool_through_SFA --num_epochs 100 --dataset_type "moving" --dataset "stl10" --patch_size 25 --tau 0.01 --sequence_length 20 --reload_BP --model_num 99 --model_path ./logs/SparsePooling_MaxPool_through_SFA_end_to_end_supervised

# 3. Test model on classification task
# TODO: implement testing only!

# 4. Potentially retrain and test linear downstream classifier with downstream_classification

# 5. Investigate learned MaxPooling layers