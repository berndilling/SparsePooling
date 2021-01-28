import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

#### own modules
from SparsePooling_pytorch.utils import logger, utils
from SparsePooling_pytorch.arg_parser import arg_parser
from SparsePooling_pytorch.models import load_model
from SparsePooling_pytorch.data import get_dataloader
from SparsePooling_pytorch.plotting import plot_weights
import matplotlib.pyplot as plt
plt.ion()

if __name__ == "__main__":
    opt = arg_parser.parse_args()
    opt.classifying = True
    
    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model = load_model.load_model(opt, reload_model=True)

    # load data
    # train_loader = get_dataloader.get_dataloader(opt)

    embed()

