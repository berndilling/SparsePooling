
using LinearAlgebra, Distributions, Statistics, ProgressMeter, JLD2, FileIO, PyPlot, MAT
include("./../sparsepooling/sparsepooling_import.jl")

## Load data
smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples = import_data("CIFAR10")
subtractmean!(smallimgs)
subtractmean!(smallimgstest)

ind = 10000 # for training & evaluating classifier

network = net(["input","sparse_patchy","pool_patchy"],
            [size(smallimgs)[1],10,20],
            [0,6,3],
            [0,1,2])

## Training
inputfunction = getsmallimg
intermediatestates = []
learn_net_layerwise!(network,intermediatestates,
    [10^4,10^3],
    [inputfunction for i in 1:network.nr_layers],
    [getstaticimage, getmovingimage];
    LearningFromLayer = 2,
    LearningUntilLayer = network.nr_layers)


if network.layer_types[end] == "classifier"
    error_train = geterrors!(network, smallimgs, labels; noftest = ind)
    error_test = geterrors!(network, smallimgstest, labelstest; noftest = ind)
    print(string("\n Train Accuracy: ", 100 * (1 - error_train)," % \n"))
    print(string("\n Test Accuracy: ", 100 * (1 - error_test)," % \n"))
else ## Train top-end classifier
    smallimgs = generatehiddenreps!(network, smallimgs;
            ind = ind, normalize = true, subtractmean = false)
    smallimgstest = generatehiddenreps!(network, smallimgstest;
            ind = ind, normalize = true, subtractmean = false)
    traintopendclassifier!(network, smallimgs, smallimgstest, labels, labelstest;
    			iters = 10^5, ind = ind, indtest = ind)
end

# TODO: Implement batch-norm like mechanism with running average?!
