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
            ax.imshow(w[i*nh+j,:,:], cmap='gray') # , vmin=-1, vmax=1)
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
    #plt.pause(0.0001)

def plot_pooled_receptive_fields(model, n_pooling_neurons = 10, n_strongest_connections = 2, select_neurons="first"):
    if len(model.module.layers) != 2:
        raise Exception("model must consist of 2 layers: SC and SFA")

    w_ff_SC = model.module.layers[0].W_ff.weight.detach().numpy() # c_post, c_pre=1, kernel_size, kernel_size
    w_ff_SFA = model.module.layers[1].W_ff.weight.detach().numpy() # c_post, c_pre, kernel_size, kernel_size
    s_SFA = w_ff_SFA.shape
    if n_pooling_neurons > s_SFA[0]:
        raise Exception("cannot extract more pooling neurons than neurons in SFA layer")
    if n_strongest_connections > s_SFA[1]:
        raise Exception("cannot extract more RFs than neurons in SC layer")
    
    if select_neurons == "rand":
        neuron_inds = torch.randint(0, s_SFA[0], (n_pooling_neurons,))
    elif select_neurons == "first":
        neuron_inds = range(n_pooling_neurons)
    w_ff_SFA_select = np.transpose(w_ff_SFA[neuron_inds, :, :, :], (0, 2, 3, 1)).reshape(n_pooling_neurons, -1) # flatten while preserving the (pre) neuron index order (dim 1); n_pooling_neurons, kernel_size*kernel_size*c_pre
    
    strongest_connections = []
    for i in range(n_pooling_neurons):
        inds_strongest_connections = w_ff_SFA_select[i,:].argsort()[-n_strongest_connections:][::-1]
        if s_SFA[-1]*s_SFA[-2] != 1: # if SFA kernel size bigger than 1x1 use modulo to solve problem that same filters appear at different positions
            inds_strongest_connections = inds_strongest_connections % (s_SFA[-1] * s_SFA[-2]) 
        strongest_connections.append(inds_strongest_connections) 

    fig, axes = plt.subplots(nrows=n_pooling_neurons, ncols=n_strongest_connections, sharex=True, sharey=True)
    fig.set_size_inches(7, 7)
    strongest_SC_rec_fields = []
    for i in range(n_pooling_neurons):
        for j in range(n_strongest_connections):
            ax = axes[i][j]
            w_plot = w_ff_SC[strongest_connections[i][j],0,:,:]
            strongest_SC_rec_fields.append(w_plot)
            ax.imshow(w_plot, cmap='gray', vmin=-1, vmax=1)
            # add normalisation?
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.figure()
    for i in range(n_pooling_neurons):
        plt.plot(w_ff_SFA_select[i, :])

    # return strongest_SC_rec_fields

# w_rec = model.module.layers[0].W_rec.squeeze().weight.detach().numpy()
# plt.figure()
# plt.imshow(w_rec,cmap='gray')
# plt.show()

# from SparsePooling_pytorch.plotting import plot_weights
# plot_weights.plot_weights(w)

