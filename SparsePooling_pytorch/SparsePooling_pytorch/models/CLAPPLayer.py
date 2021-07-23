from os import EX_CANTCREAT
from IPython.core.interactiveshell import ExecutionInfo
import torch.nn as nn
import torch
import numpy as np
from IPython import embed


class CPCLayer(nn.Module):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p, stride=1, padding=0, do_update_params = True): # stride=max(kernel_size//2, 1)
        super(CPCLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.W_ff = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.nonlin = nn.ReLU(inplace=False) 

        self.update_params = do_update_params

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.learning_rate)

    def forward(self, input):
        self.optimizer.zero_grad()

        u = self.W_ff(input) # b, c, x, y
        a = self.nonlin(u) # b, c, x, y
        return a
    
    def update_parameters(self, post):
        if self.update_params:
            self.optimizer.step()

# layer for end-to-end CLAPP (HingeLossCPC)
class HingeCPCLayer(CPCLayer):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p, stride=1, padding=0, do_update_params = True): 
        super(HingeCPCLayer, self).__init__(opt, in_channels, out_channels, kernel_size, p, stride=stride, padding=padding, do_update_params=do_update_params)

        self.W_ff = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.nonlin = nn.ReLU(inplace=False)


class CLAPPLayer(CPCLayer):
    def __init__(self, opt, in_channels, out_channels, kernel_size, p, stride=1, padding=0, do_update_params = True): 
        super(CLAPPLayer, self).__init__(opt, in_channels, out_channels, kernel_size, p, stride=stride, padding=padding, do_update_params=do_update_params)

        self.Loss = CLAPPLoss(opt, out_channels)

    # overwrite parent's method 
    def update_parameters(self, post):
        if self.update_params:
            self.zero_grad()
            loss = self.loss(post)
            loss.backward()
            self.optimizer.step()


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        
    def forward(self, input, positive):
        # positive IS USED TO KNOW THE POSITIVES IN THE SCORES MATRIX (IN OUR CASE, IT'S THE DIAGONAL)
        input[positive, positive] *= -1
        # APPLYING THE MAX(1+INPUT,0)
        input = torch.clamp(1+input, min=0)
        # SUMMATION OF THE 2 LOSS COMPONENTS 

        # TODO implement either or (pos, neg) functionality
        loss = 0.5*(torch.mean(input[positive, positive]) + (torch.sum(input)- torch.sum(input[positive, positive]))/((positive.size(0)-1)*(positive.size(0))))
        return loss


class CLAPPLoss(nn.Module):
    def __init__(self, opt, out_channels):
        self.loss = HingeLoss()

        # TODO prediction weights

    def forward():

        # implement actual CLAPP/DPC... see Jean's impl.

        # predictions

        # negatives

        # contrasting + self.loss(...)

        return
