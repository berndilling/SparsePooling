
using MAT
using HDF5
using JLD
using PyPlot

include("sparsepooling_import.jl")

#dataset to be used
dataset = "Olshausen_white"#"MNIST144"
labelled = false

iterations = 10^5

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
network = net([256,300],["input","sparse"])

learn_layer_sparse!(network.layers[1], network.layers[2], getsmallimg, iterations)

save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/tests/test_sparse_olshausen.jld", "network", network)
