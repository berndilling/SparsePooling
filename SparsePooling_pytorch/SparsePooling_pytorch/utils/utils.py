# Taken from:
# https://github.com/loeweX/Greedy_InfoMax

import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_device(opt, input_tensor):
    if opt.device.type != "cpu":
        cur_device = input_tensor.get_device()
    else:
        cur_device = opt.device

    return cur_device

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def getsparsity(x:torch.tensor):
    return float(torch.sum(x == 0))/np.prod(x.shape)

# data is (b, c) array of b c-dimensional vectors
def subtractmean(data):
    return data - torch.mean(data, (0))

# data is (b, c) array of b c-dimensional vectors
def whiten(data):
    data = subtractmean(data)
    U, S, V = torch.svd(data)
    return torch.matmul(U, V.t()) * np.sqrt(data.shape[0] - 1)
