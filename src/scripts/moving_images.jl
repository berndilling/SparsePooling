
using LinearAlgebra, Statistics, StatsBase, ProgressMeter, JLD2, FileIO, PyPlot, MAT
include("./../sparsepooling/sparsepooling_import].jl")

## Load data
smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples = import_data("CIFAR10")
subtractmean!(smallimgs)
subtractmean!(smallimgstest)

## Create network
network = net([size(smallimgs)[1],10,10], #20, 10,20,10
            ["input","sparse_patchy","classifier"]; #  "pool_patchy","sparse_patchy","pool_patchy"
            overlap = true)

## Training
inputfunction = getsmallimg
dynamicfunction =  getstaticimage #getmovingimage #

intermediatestates = []
learn_net_layerwise!(network,intermediatestates,[10^4,10^4],#[10^3 for i in 1:network.nr_layers-1],
  [inputfunction for i in 1:network.nr_layers-1],
  [dynamicfunction for i in 1:network.nr_layers-1];
  LearningFromLayer = 2, LearningUntilLayer = network.nr_layers)

noftest = 10^3 #!!!
error_train = geterrors!(network, smallimgs, labels; noftest = noftest)
error_test = geterrors!(network, smallimgstest, labelstest; noftest = noftest)
print(string("Train Error: ", 100 * error_train," %"))
print(string("Test Error: ", 100 * error_test," %"))

# TODO: Implement batch-norm like mechanism with running average?!
