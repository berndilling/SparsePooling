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

def train(opt, model, train_loader, logs):
    model.train()
    if not model.module.update_params:
        raise Exception("Cannot train model if model plasticity switched of (.update_params = False")
    if opt.train_layer != None:
        for idx, layer in enumerate(model.module.layers):
            layer.update_params = (idx == opt.train_layer)
        print("only training layer number ", opt.train_layer)
    
    logs.create_log(model, epoch=-1)
    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch + 1):   
        print("epoch ", epoch, " of ", opt.num_epochs)
        save_first_layer_weights(opt, model, epoch)
        
        sparsity = 0
        CLAPPLoss = []
        for step, (img) in enumerate(train_loader):
            input = img.to(opt.device)
            out, loss = model(input, up_to_layer = opt.train_layer) # param updates happen online
            if step % 100 == 0:
               print("batch number: ", step, " out of ", len(train_loader))
            sparsity += utils.getsparsity(out)
            CLAPPLoss.append(loss)

        save_loss(epoch, model, CLAPPLoss)
            
        print("epoch average sparsity of last layer: ", sparsity / len(train_loader))
        logs.create_log(model, epoch=epoch)
    
    plt.show()

def save_first_layer_weights(opt, model, epoch):
    if epoch % (opt.num_epochs // min(opt.num_epochs, 10)) == 0:
        plot_weights.plot_receptive_fields(model.module.layers[0].W_ff.weight.clone().detach(), nh=10, nv=10) # nh=20, nv=20)
        plt.savefig(os.path.join(opt.log_path,'W_ff'+str(epoch)+'.png'))

def save_loss(epoch, model, CLAPPLoss):
    if model.module.do_loss:
        # calculate average loss over epoch and save it
        for i in range(len(CLAPPLoss[0])): # for all layers
            l = 0
            for b in range(len(CLAPPLoss)):
                l += CLAPPLoss[b][i]

            l /= len(CLAPPLoss)
            
            print("epoch average CLAPP loss for (CLAPP-)layer ",str(i),": ",str(l.item()))
            L = [str(l.item())+'\n']
            if epoch == 0:
                write_mode = "w"
            else:
                write_mode = "a"
            f = open(os.path.join(opt.model_path, "logs", opt.save_dir, "CLAPP_loss_CLAPPlayer_"+str(i)+".txt"), write_mode)
            f.writelines(L)
            f.close()

if __name__ == "__main__":
    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    
    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model = load_model.load_model(opt)
    model.module.set_update_params(update_model = True, update_BP = False, update_SC_SFA = True)

    # load data
    train_loader = get_dataloader.get_dataloader(opt)

    # create logger
    logs = logger.Logger(opt)

    # train model
    try:
        # Train the model
        train(opt, model, train_loader, logs)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")
    
    logs.create_log(model)