
using MAT
using HDF5
using JLD
using PyPlot

include("sparsepooling_import.jl")

#dataset to be used
dataset = "CIFAR10"#"MNIST144_white"#"Olshausen_white"#"CIFAR10_whitened"#"Olshausen_white"#"MNIST144"
labelled = true#false

iterations = 10^6

if labelled
  smallimgs, labels, smallimgstest, labelstest, n_samples, n_testsamples =  import_data(dataset)
else
  smallimgs, n_samples = import_unlabelled_data(dataset)
end

#THIS MIGHT NOT EVEN BE HELPFUL!
#substract line-wise (pixel/variable-wise) mean
if labelled
  subtractmean!(smallimgs)
  subtractmean!(smallimgstest)
else
  subtractmean!(smallimgs)
end

#Create network with two layers: ordinary sparse coding setup
network = net([size(smallimgs)[1],10],["input","sparse"])

learn_layer_sparse!(network.layers[1], network.layers[2], getsmallimg, iterations)
generatehiddenreps(network.layers[1], network.layers[2])

save(string(getsavepath(),"SparsePooling/analysis/tests/sparse_test_weight_decay_",dataset,".jld"), "network", network)
