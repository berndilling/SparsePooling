import matplotlib.pyplot as plt
import numpy as np
import torch 

def plot_responses_bars(model, size=10):
    X = torch.zeros(2 * size, 1, size, size) # n_img, n_channels, img_size, img_size
    for i in range(size):
        X[i, 0, i, :] = 1. # horizontal bars
        X[i+size, 0, :, i] = 1. # vertical bars

    out_SC, _ = model.module(X, up_to_layer=0)
    out_SFA, _ = model.module(X, up_to_layer=1)
    plt.figure()
    plt.imshow(out_SC[:,:,0,0].t().detach())
    plt.xlabel('pattern #')
    plt.ylabel('SC neuron index')
    plt.figure()
    plt.imshow(out_SFA[:,:,0,0].t().detach())
    plt.xlabel('pattern #')
    plt.ylabel('SFA neuron index')
    plt.figure()
    plt.plot(out_SFA[:,:,0,0].detach())
    plt.xlabel('pattern #')
    plt.ylabel('SFA response')
