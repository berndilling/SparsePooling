# Taken from:
# https://github.com/loeweX/Greedy_InfoMax

from optparse import OptionParser
import time
import os
import torch
import numpy as np

from SparsePooling_pytorch.arg_parser import args


def parse_args():
    # load parameters and options
    parser = OptionParser()

    parser = args.parse_args(parser)
    
    (opt, _) = parser.parse_args()

    opt.time = time.ctime()

    # Device configuration
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt.experiment = "vision"

    return opt


def create_log_path(opt, add_path_var=""):
    unique_path = False

    if opt.save_dir != "":
        opt.log_path = os.path.join(opt.data_output_dir, "logs", opt.save_dir)
        unique_path = True
    elif add_path_var == "features" or add_path_var == "images":
        opt.log_path = os.path.join(opt.data_output_dir, "logs", add_path_var, os.path.basename(opt.model_path))
        unique_path = True
    else:
        opt.log_path = os.path.join(opt.data_output_dir, "logs", add_path_var, opt.time)

    # hacky way to avoid overwriting results of experiments when they start at exactly the same time
    while os.path.exists(opt.log_path) and not unique_path:
        opt.log_path += "_" + str(np.random.randint(100))

    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)

