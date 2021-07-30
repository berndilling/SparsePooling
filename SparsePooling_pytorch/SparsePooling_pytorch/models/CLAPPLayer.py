from os import EX_CANTCREAT
from IPython.core.interactiveshell import ExecutionInfo
import torch.nn as nn
import torch
import numpy as np
from IPython import embed

torch.autograd.set_detect_anomaly(True)

class CPCLayer(nn.Module):
    def __init__(self, opt, in_channels, out_channels, kernel_size, stride=1, padding=0, do_update_params = True): # stride=max(kernel_size//2, 1)
        super(CPCLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.seq_length = opt.sequence_length

        self.W_ff = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.nonlin = nn.ReLU(inplace=False) 

        self.update_params = do_update_params

    def forward(self, input):
        u = self.W_ff(input) # b, c, x, y
        a = self.nonlin(u) # b, c, x, y
        return a
    
    def update_parameters(self):
        if self.update_params:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return None

# layer for end-to-end CLAPP (HingeLossCPC)
class HingeCPCLayer(CPCLayer):
    def __init__(self, opt, in_channels, out_channels, kernel_size, stride=1, padding=0, do_update_params = True): 
        super(HingeCPCLayer, self).__init__(opt, in_channels, out_channels, kernel_size, stride=stride, padding=padding, do_update_params=do_update_params)

        self.W_ff = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.nonlin = nn.ReLU(inplace=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.learning_rate)

class CLAPPLayer(CPCLayer):
    def __init__(self, opt, in_channels, out_channels, kernel_size, stride=1, padding=0, do_update_params = True): 
        super(CLAPPLayer, self).__init__(opt, in_channels, out_channels, kernel_size, stride=stride, padding=padding, do_update_params=do_update_params)

        self.Loss = CLAPPLoss(opt, out_channels)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.learning_rate)

    # overwrite parent's method 
    def update_parameters(self, post):
        if self.update_params:
            self.optimizer.zero_grad()
            loss = self.Loss(post)
            loss.backward()
            self.optimizer.step()
        
        return loss.clone().detach()

class CLAPPLoss(nn.Module):
    def __init__(self, opt, out_channels, n_predictions = 3, n_negatives=1):
        super(CLAPPLoss, self).__init__()
        self.n_preds = n_predictions
        self.n_spatial_pool_patches = opt.n_spatial_pool_patches
        self.do_spatial_pool = self.n_spatial_pool_patches != 0
        self.n_negatives = n_negatives
        self.seq_length = opt.sequence_length

        self.spatial_pool = nn.AdaptiveAvgPool2d((self.n_spatial_pool_patches, self.n_spatial_pool_patches)) # spatial pooling (resembles patch encoding in image CPC/CLAPP, but without overlapping patches)

        self.loss = HingeLoss(async_update=opt.async_update)

        self.Wpred = nn.ModuleList() 
        for i in range(self.n_preds): # number of pred_steps
            self.Wpred.append(nn.Conv2d(out_channels, out_channels, kernel_size=(1,1), bias=False))     

    def forward(self, input):
        # input: b' (=b*sequence_length), c, x, y
        if self.do_spatial_pool:
            input = self.spatial_pool(input) # spatial pooling (resembles patch encoding in image CPC/CLAPP)
        
        s = input.shape

        input = input.reshape(-1, self.seq_length, s[1], s[2], s[3]) # b, sequence_length, c, x, y

        loss = 0
        for i in range(self.n_preds):
            # define context (basis for prediction)
            context = input[:, :-(i+1), :, :, :] # first seq_length-(i+1) frames: b, seq_length-(i+1), c, x, y

            # prediction
            pred = self.Wpred[i](context.reshape(-1, s[1], s[2], s[3])).reshape(-1, self.seq_length-(i+1), s[1], s[2], s[3]) # b, seq_length-(i+1), c, x, y

            # target
            target = input[:, (i+1):, :, : :] # last seq_length-(i+1) frames: b, seq_length-(i+1), c, x, y

            # negatives
            if self.n_negatives == 1:
                rand_index = torch.randint(target.shape[0], (target.shape[0],), dtype=torch.long, device=target.get_device()) # upper limit b, shape: b, assumes n=1 neg. samples, 
                negatives = target[rand_index, :, :, :, :] # same as target: b, seq_length-(i+1), c, x, y
            else: # TODO implement higher number of negatives
                raise NotImplementedError("Not implemented yet!")# select self.n_negatives 

            # contrasting
            scores_pos = torch.einsum('bscxy, bscxy -> bsxy', pred, target) / s[1] # b, seq_length-(i+1), x, y
            scores_neg = torch.einsum('bscxy, bscxy -> bsxy' , pred, negatives) / s[1] # b, seq_length-(i+1), x, y

            loss += self.loss(scores_pos, scores_neg)

        return loss / self.n_preds

class HingeLoss(nn.Module):
    def __init__(self, async_update=False):
        super(HingeLoss, self).__init__()
        self.async_update = async_update
        
    def forward(self, scores_pos, scores_neg):
        loss_pos = torch.clamp(1 - scores_pos, min=0) # b, (red. seq_length), x, y
        loss_neg = torch.clamp(1 + scores_neg, min=0)

        loss_pos = loss_pos.mean(dim=(-1,-2,-3)) # b
        loss_neg = loss_neg.mean(dim=(-1,-2,-3)) # b
 
        if self.async_update:
            loss = 0
            which_update = np.random.choice(['pos','neg'], size = loss_pos.shape[0], replace=True, p = [0.5,0.5]) # b
            for Loss, c in zip([loss_pos, loss_neg], ['pos', 'neg',]):
                    ind = (which_update == c)
                    if sum(ind) > 0: # exclude empty sets which lead no NaN in loss
                        loss += torch.masked_select(Loss, torch.tensor(ind.tolist()).to(scores_pos.get_device())).mean()
        else:
            loss = loss_pos.mean() + loss_neg.mean()

        return loss
