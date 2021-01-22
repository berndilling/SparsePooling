
import torch
import time
import numpy as np
from IPython import embed

#### own modules
from SparsePooling_pytorch.utils import logger, utils
from SparsePooling_pytorch.arg_parser import arg_parser
from SparsePooling_pytorch.models import load_model
from SparsePooling_pytorch.data import get_dataloader


def train(opt, model, train_loader, logs):
    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):
        print("epoch ", epoch, " of ", opt.num_epochs-1)
        for step, (img) in enumerate(train_loader):
            input = img.to(opt.device)
            out = model(input)

        print("Sparsity: ", utils.getsparsity(out))
        logs.create_log(model, epoch=epoch)


if __name__ == "__main__":
    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    
    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model = load_model.load_model(opt)

    # load data
    train_loader = get_dataloader.get_dataloader(opt)

    # create logger
    logs = logger.Logger(opt)

    # train model
    try:
        # Train the model
        train(opt, model, train_loader, logs)
        embed()

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")
    
    logs.create_log(model)