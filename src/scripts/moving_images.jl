
using LinearAlgebra, Distributions, Statistics, ProgressMeter, JLD2, FileIO, PyPlot, MAT
include("./../sparsepooling/sparsepooling_import.jl")

## Load data
smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples = import_data("CIFAR10")
subtractmean!(smallimgs)
subtractmean!(smallimgstest)

## Create network
# network = net(["input","sparse_patchy","pool_patchy","sparse_patchy"],
#             [size(smallimgs)[1],10,5,10],
#             [0,10,10,10],
#             [0,1,1,1])
network = net(["input","sparse_patchy"],
            [size(smallimgs)[1],10],
            [0,10],
            [0,2])

## Training
inputfunction = getsmallimg
intermediatestates = []
learn_net_layerwise!(network,intermediatestates,
    [10^3],
    [inputfunction for i in 1:network.nr_layers],
    [getstaticimage, getmovingimage, getstaticimage];
    LearningFromLayer = 2,
    LearningUntilLayer = network.nr_layers)

## Train top-end classifier
ind = 1000
reps = generatehiddenreps!(network, smallimgs;
        ind = ind, normalize = true, subtractmean = false)
repstest = generatehiddenreps!(network, smallimgstest;
        ind = ind, normalize = true, subtractmean = false)

smallimgs = reps
smallimgstest = repstest

traintopendclassifier!(network, smallimgs, smallimgstest, labels, labelstest;
			iters = 10^5, ind = ind, indtest = ind)

# TODO: Implement batch-norm like mechanism with running average?!
