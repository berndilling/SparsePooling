import matplotlib.pyplot as plt
import numpy as np
import torch 

def plot_weights(weights, k=10, nh=8, nv=4):
    w = weights.squeeze()
    im = torch.zeros(k*nv, k*nh)
    for i in range(nv):
        for j in range(nh):
            im[i*k:(i+1)*k,j*k:(j+1)*k] = w[i*nh+j,:,:]

    return im.detach().numpy() 

w = model.module.layers[0].W_ff.weight
im = plot_weights(w, nh=10, nv=10)
plt.figure()
plt.imshow(im,cmap='gray')
plt.show()

# from SparsePooling_pytorch.plotting import plot_weights
# plot_weights.plot_weights(w)