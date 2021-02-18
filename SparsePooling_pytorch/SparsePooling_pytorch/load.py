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
from SparsePooling_pytorch.plotting import plot_activity
import matplotlib.pyplot as plt

if __name__ == "__main__":
    opt = arg_parser.parse_args()
    opt.classifying = True
    
    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model = load_model.load_model(opt, reload_model=True)

    # load data
    train_loader = get_dataloader.get_dataloader(opt)

    plt.ion()
    embed()

# plt.ion()
# ims = train_loader.dataset.images
# out, _ = model(ims[0:1000, :, : ,: ], up_to_layer = ?)
# plt.plot(torch.mean(out, (0,2,3)))
