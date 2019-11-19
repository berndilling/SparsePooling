using Pkg; Pkg.activate("./../SparsePooling/"); Pkg.instantiate()
push!(LOAD_PATH, "./../SparsePooling/src/")
using SparsePooling
using PyPlot

## Load data
smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples = import_data("CIFAR10")
subtractmean!(smallimgs)
subtractmean!(smallimgstest)
data = labelleddata(smallimgs, labels)
datatest = labelleddata(smallimgstest, labelstest)

ind = 50000 # for training & evaluating classifier
ind_t = 10000

network = net(["input","sparse_patchy","pool_patchy"],
            [size(data.data)[1],10,10],
            [0,6,3],
            [0,1,2])

## Training
inputfunction = getsmallimg
intermediatestates = []
learn_net_layerwise!(network, data, intermediatestates,
    [10^4,10^3],
    [inputfunction for i in 1:network.nr_layers],
    [getstaticimage, getmovingimage];
    LearningFromLayer = 2,
    LearningUntilLayer = network.nr_layers)


# if network.layer_types[end] == "classifier"
#     error_train = geterrors!(network, data; noftest = ind)
#     error_test = geterrors!(network, datatest; noftest = ind_t)
#     print(string("\n Train Accuracy: ", 100 * (1 - error_train)," % \n"))
#     print(string("\n Test Accuracy: ", 100 * (1 - error_test)," % \n"))
#else ## Train top-end classifier
    lasthiddenrepstrain = labelleddata(generatehiddenreps!(network, data;
            ind = ind, normalize = true, subtractmean = false), data.labels[1:ind])
    lasthiddenrepstest = labelleddata(generatehiddenreps!(network, datatest;
            ind = ind_t, normalize = true, subtractmean = false), datatest.labels[1:ind_t])
    traintopendclassifier!(network, lasthiddenrepstrain, lasthiddenrepstest;
    			iters = 10^6, ind = ind, indtest = ind_t)
#end

# TODO: Implement batch-norm like mechanism with running average?!
