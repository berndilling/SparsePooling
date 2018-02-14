
using StatsBase, ProgressMeter, JLD, PyPlot
close("all")
include("./sparsepooling/sparsepooling_import.jl")

sparse_part = true
pool_part = false

iterations = 10^5#5
in_size = 64
hidden_size = 16#32

if sparse_part
  network = net([in_size,hidden_size],["input","sparse_patchy","pool"])
  for i in 1:network.layers[2].parameters.n_of_sparse_layer_patches
    set_init_bars!(network.layers[2].sparse_layer_patches[i],hidden_size)
    network.layers[2].sparse_layer_patches[i].parameters.p = 1/(16)
  end
  #network.layers[2].parameters.activationfunction = "relu"

  learn_layer_sparse_patchy!(network.layers[1], network.layers[2], iterations)
  #generatehiddenreps(network.layers[1], network.layers[2], smallimgs; number_of_reps = 10^4)

  save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_test.jld","network",network)

end
if pool_part
  network = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_test.jld","network")

  #learn_layer_pool_patchy!(network.layers)

end
