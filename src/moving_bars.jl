using MAT
using HDF5
using JLD
using PyPlot

include("sparsepooling_import.jl")

sparse_part = false
pool_part = true

#dataset to be used
dataset = "bars"
labelled = false

iterations = 10^6

smallimgs, n_samples = import_unlabelled_data(dataset)

#scale data between [-1,1]
#rescaledata!(smallimgs)

#THIS MIGHT NOT EVEN BE HELPFUL!
#substract line-wise (pixel/variable-wise) mean
subtractmean!(smallimgs)

if sparse_part
  #Create network with two layers: ordinary sparse coding setup
  network = net([size(smallimgs)[1],200],["input","sparse"])

  errors, ffd = learn_layer_sparse!(network.layers[1], network.layers[2], getsmallimg, iterations)
  generatehiddenreps(network.layers[1], network.layers[2], number_of_reps = 24)

  save(string(getsavepath(),"SparsePooling/analysis/tests/sparse_",dataset,".jld"),
      "network", network, "squared_errors", errors, "ffd", ffd)
end
if pool_part
  network = load(string(getsavepath(),"SparsePooling/analysis/tests/sparse_bars_",dataset,".jld"),"network")
  smallimgs = network.layers[2].hidden_reps
  network_2 = net(net([200,1],["input","pool"]))
  errors = learn_layer_pool!(network_2.layers[1], network_2.layers[2], get_moving_vbar, iterations)
  generatehiddenreps(network_2.layers[1], network_2.layers[2], number_of_reps = 24)

  save(string(getsavepath(),"SparsePooling/analysis/tests/pool_",dataset,".jld"),
      "network", network_2, "squared_errors", errors)
end
