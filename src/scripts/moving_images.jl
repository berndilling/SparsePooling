
using StatsBase, ProgressMeter, JLD2, FileIO, PyPlot, MAT
include("./../sparsepooling/sparsepooling_import.jl")

## Load data
smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples = import_data("CIFAR10")

## Create network
network = net([size(smallimgs)[1],20,10,30,15,10],
            ["input","sparse_patchy","pool_patchy","sparse_patchy","pool_patchy","classifier"];
            overlap = true)

## Training
inputfunction = getsmallimg
dynamicfunction = getmovingimage

intermediatestates = []
learn_net_layerwise!(network,intermediatestates,[10^1 for i in 1:network.nr_layers-1],
  [inputfunction for i in 1:network.nr_layers-1],
  [dynamicfunction for i in 1:network.nr_layers-1])

error_train = geterrors!(network, smallimgs, labels)
error_test = geterrors!(network, smallimgstest, labelstest)

# TODO: Implement batch-norm like mechanism with running average?!
