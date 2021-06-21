import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tvF

def plot_responses_bars(model, size=10):
    X = torch.zeros(2 * size, 1, size, size) # n_img, n_channels, img_size, img_size
    for i in range(size):
        X[i, 0, i, :] = 1. # horizontal bars
        X[i+size, 0, :, i] = 1. # vertical bars

    out_SC, _ = model.module(X, up_to_layer=0)
    out_SFA, _ = model.module(X, up_to_layer=1)
    plt.figure()
    plt.imshow(out_SC[:,:,0,0].t().detach(), cmap = 'gray')
    plt.xlabel('pattern #')
    plt.ylabel('SC neuron index')
    plt.figure()
    plt.imshow(out_SFA[:,:,0,0].t().detach(), cmap = 'gray')
    plt.xlabel('pattern #')
    plt.ylabel('SFA neuron index')
    plt.figure()
    plt.plot(out_SFA[:,:,0,0].detach())
    plt.xlabel('pattern #')
    plt.ylabel('SFA response')

def plot_simple_and_complex_response(model, size=10, stim_strength = 5, rot = False):
    X = torch.zeros(4 * size, 3, size, size) # n_img, n_channels, img_size, img_size
    for i in range(size):
        X[i, 0, i, :] = stim_strength # horizontal bars
        X[i+size, 0, :, i] = stim_strength # vertical bars
        for j in range(-np.abs(i-size//2)+size): # diag bars
            if i <= 5:
                X[i+2*size, 0, min(i+size//2-1-j,9), j] = stim_strength 
            if i > 5:
                X[i+2*size, 0, size-j-1, min(i-size//2+j,9)] = stim_strength
        X[i+3*size, 0, :, :] = torch.flip(X[i+2*size, 0, :, :], (0,)) # diag bars
    
    if rot: # manual extra rotation to maximally activate diagonals
        X_rot = torch.zeros(4 * size, 3, size, size) # n_img, n_channels, img_size, img_size
        X_rot[:size,:,:,:] = tvF.rotate(X[:size,:,:,:], 25) # 65
        X_rot[size:2*size, :, :, :] = tvF.rotate(X[:size,:,:,:], -60) # -60
        X_rot[2*size:, :, :, :] = X[2*size:, :, :, :]
        input = X_rot[:, 0, :, :].unsqueeze(1)
    else:    
        input = X[:, 0, :, :].unsqueeze(1)
    

    out_SC, _ = model.module(input, up_to_layer=0)
    out_SFA, _ = model.module(input, up_to_layer=1)

    fig, axes = plt.subplots(2,1, sharex=True)
    
    ax = axes[0]
    #fig = plt.figure()
    #ax = fig.gca()
    ax.imshow(out_SC[:40,:100,0,0].t().detach(), cmap = 'gray')
    ax.set_xlabel('pattern #')
    ax.set_ylabel('SC neuron index')
    ax.set_aspect(.1)

    ax = axes[1]
    #fig = plt.figure()
    #ax = fig.gca()
    ax.imshow(out_SFA[:40,:,0,0].t().detach(), cmap = 'gray')
    ax.set_xlabel('pattern #')
    ax.set_ylabel('SFA neuron index')
    #ax.set_aspect(3.)

    #ax = axes[2]
    fig, axes = plt.subplots(4,1, sharex=True)
    for i in range(4):
        ax = axes[i]
        ax.imshow(input[i*size:(i+1)*size, 0, :, :].permute(0,2,1).reshape(size*size, size).t(), cmap = 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.suptitle('input sequence')