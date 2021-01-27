import matplotlib.pyplot as plt
import numpy as np
import torch 

def plot_receptive_fields(weights, nh=10, nv=10):
    w = weights.cpu().squeeze().detach().numpy()
    fig, axes = plt.subplots(nrows=nv, ncols=nh, sharex=True, sharey=True)
    fig.set_size_inches(7, 7)
    for i in range(nv):
        for j in range(nh):
            ax = axes[i][j]
            ax.imshow(w[i*nh+j,:,:], cmap='gray', vmin=-1, vmax=1)
            # add normalisation?
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()
    # plt.pause(0.0001)

def plot_weights(weights, k=10, nh=10, nv=10):
    w = weights.cpu().squeeze()
    im = torch.zeros(k*nv, k*nh)
    for i in range(nv):
        for j in range(nh):
            im[i*k:(i+1)*k,j*k:(j+1)*k] = w[i*nh+j,:,:]

    plt.imshow(im.detach().numpy(),cmap='gray')
    plt.show()
    plt.pause(0.0001)
    #return im.detach().numpy() 

# w = model.module.layers[0].W_ff.weight
# im = plot_weights(w, nh=10, nv=10)
# plt.figure()
# plt.imshow(im,cmap='gray')
# plt.show()

# w_rec = model.module.layers[0].W_rec.weight.squeeze().detach().numpy()
# plt.figure()
# plt.imshow(w_rec,cmap='gray')
# plt.show()

# from SparsePooling_pytorch.plotting import plot_weights
# plot_weights.plot_weights(w)