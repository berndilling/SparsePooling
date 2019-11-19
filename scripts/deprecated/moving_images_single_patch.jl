using LinearAlgebra, Distributions, Statistics, ProgressMeter, JLD2, FileIO, PyPlot, MAT
include("./../sparsepooling/sparsepooling_import.jl")

# IDEA 1 (implemented right now): create minimal "pyramid" up to a certain layer and train it (using image patches)
# Then use weight sharing to initialize model for training higher layers!
# IDEA 2: Create hierarchical patch-wise training!
# -> train first layer, glue together trained patch for second layer, train second layer patch etc

## Load data
smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples = import_data("CIFAR10")
subtractmean!(smallimgs)
subtractmean!(smallimgstest)

## Play with lc_forward = false?!

patch_size = 7
network = net(["input","sparse_patchy","pool_patchy"],
            [patch_size^2,10,10],
            [0,5,3],
            [0,1,1])

## Training
intermediatestates = []
losses = learn_net_layerwise!(network,intermediatestates, [10^5, 10^1],
                    [getsmallimg,getsmallimg],
                    [getstaticimagepatch, getmovingimagepatch],
                    cut_size = patch_size, eval_loss = true)

# TODO: code function for saving and loading learned layers into big network with weight sharing
