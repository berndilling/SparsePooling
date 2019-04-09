
using LinearAlgebra, Distributions, Statistics, ProgressMeter, JLD2, FileIO, PyPlot, MAT
include("./../sparsepooling/sparsepooling_import.jl")

## Load data
smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples = import_data("CIFAR10")
subtractmean!(smallimgs)
subtractmean!(smallimgstest)

## Create network
network = net([size(smallimgs)[1],10,5,10], #20, 10,20,10
            ["input","sparse_patchy","pool_patchy","sparse_patchy"], #  "pool_patchy","sparse_patchy","pool_patchy"
            [0,10,10,10], # patch/kernel sizes
            [0,1,1,1]) # strides

## Training
inputfunction = getsmallimg
dynamicfunction =  getstaticimage #getmovingimage #

intermediatestates = []
learn_net_layerwise!(network,intermediatestates,[10^3,10^2,10^3],#[10^3 for i in 1:network.nr_layers-1],
  [inputfunction for i in 1:network.nr_layers],
  [getstaticimage, getmovingimage, getstaticimage]; # vcat([dynamicfunction for i in 1:network.nr_layers-2],getstaticimage);
  LearningFromLayer = 2, LearningUntilLayer = network.nr_layers)

ind = 1000#0
class1 = net([length(network.layers[network.nr_layers].a),10], ["input","classifier"])
reps = zeros(length(network.layers[network.nr_layers].a),ind)
repstest = zeros(length(network.layers[network.nr_layers].a),ind)
@info("calculate hidden reps")
@showprogress for i in 1:ind # size(smallimgs)[2]
    network.layers[1].a = smallimgs[:,i]
    forwardprop!(network, FPUntilLayer = network.nr_layers)
    reps[:,i] = deepcopy(network.layers[network.nr_layers].a)
end
maximum(reps) != 0 && (reps ./= maximum(reps))
smallimgs = deepcopy(reps)

@showprogress for i in 1:ind # size(smallimgstest)[2]
    network.layers[1].a = smallimgstest[:,i]
    forwardprop!(network, FPUntilLayer = network.nr_layers)
    repstest[:,i] = deepcopy(network.layers[network.nr_layers].a)
end
maximum(repstest) != 0 && (repstest ./= maximum(repstest))
smallimgstest = deepcopy(repstest)

i2 = []
learn_net_layerwise!(class1,i2,[10^6],
  [inputfunction for i in 1:network.nr_layers-1],
  [getstatichiddenrep for i in 1:network.nr_layers-1];
  LearningFromLayer = 2, LearningUntilLayer = 2)


noftest = ind # 10^4 #!!!
error_train = geterrors!(class1, smallimgs, labels; noftest = noftest)
error_test = geterrors!(class1, smallimgstest, labelstest; noftest = noftest)
print(string("\n Train Accuracy: ", 100 * (1 - error_train)," % \n"))
print(string("\n Test Accuracy: ", 100 * (1 - error_test)," % \n"))

# TODO: Implement batch-norm like mechanism with running average?!
