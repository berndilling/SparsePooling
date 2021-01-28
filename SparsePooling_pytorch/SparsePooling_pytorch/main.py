import os
import torch
import time
import numpy as np
from IPython import embed

#### own modules
from SparsePooling_pytorch.utils import logger, utils
from SparsePooling_pytorch.arg_parser import arg_parser
from SparsePooling_pytorch.models import load_model
from SparsePooling_pytorch.data import get_dataloader
from SparsePooling_pytorch.plotting import plot_weights
import matplotlib.pyplot as plt
plt.ion()

# TODO: use torch.optim.lr_scheduler!
def decay_learningrates(model, opt):
    for layer in model.module.layers:
        for g in layer.optimizer.param_groups:
            g['lr'] *= opt.learning_rate_decay
            # print(g['lr'])

def train(opt, model, train_loader, logs):
    logs.create_log(model, epoch=-1)
    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch + 1):
        print("epoch ", epoch, " of ", opt.num_epochs)
        if epoch % (opt.num_epochs // min(opt.num_epochs, 10)) == 0:
            plot_weights.plot_receptive_fields(model.module.layers[0].W_ff.weight.clone().detach())
            plt.savefig(os.path.join(opt.log_path,'W_ff'+str(epoch)+'.png'))
        for step, (img) in enumerate(train_loader):
            input = img.to(opt.device)
            out, dparams = model(input)
            # if step % 100 == 0:
            #    print(step)
                # print("Change in W_ff :", torch.norm(dparams[0]).data)
            
        # TODO: implement SC loss and keep track it
        print("Sparsity: ", utils.getsparsity(out))
        decay_learningrates(model, opt)

        logs.create_log(model, epoch=epoch)
    
    plt.show()


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
        # embed()

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")
    
    logs.create_log(model)