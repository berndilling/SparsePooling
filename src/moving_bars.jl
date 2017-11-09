using MAT
using HDF5
using JLD
using PyPlot

include("sparsepooling_import.jl")

sparse_part = true
pool_part = true

#dataset to be used
dataset_sparse = "bars_superimposed"
dataset_pool = "bars"
labelled = false

hidden_size = 72

iterations = 10^5

smallimgs_sparse, n_samples = import_unlabelled_data(dataset_sparse)
smallimgs_pool, n_samples = import_unlabelled_data(dataset_pool)
#scale data between [-1,1]
#rescaledata!(smallimgs)

#THIS MIGHT NOT EVEN BE HELPFUL!
#substract line-wise (pixel/variable-wise) mean
subtractmean!(smallimgs_sparse)
subtractmean!(smallimgs_pool)

if sparse_part
  smallimgs = smallimgs_sparse
  #Create network with two layers: ordinary sparse coding setup
  network = net([size(smallimgs)[1],hidden_size],["input","sparse"])
  #set initialization appropriate to bar-data
  set_init_bars!(network.layers[2],hidden_size)
  # only learn thresholds to reach stable values
  errors, ffd = learn_layer_sparse!(network.layers[1], network.layers[2], getsmallimg, 10^3, lr_v = 0., lr_w = 0., lr_thr = 1e-1)
  errors, ffd = learn_layer_sparse!(network.layers[1], network.layers[2], getsmallimg, iterations)
  generatehiddenreps(network.layers[1], network.layers[2], number_of_reps = size(smallimgs)[2])

  save(string(getsavepath(),"SparsePooling/analysis/tests/sparse_",dataset_sparse,".jld"),
      "network", network, "squared_errors", errors, "ffd", ffd)
end
if pool_part
  network = load(string(getsavepath(),"SparsePooling/analysis/tests/sparse_",dataset_sparse,".jld"),"network")
  smallimgs = smallimgs_pool
  generatehiddenreps(network.layers[1], network.layers[2], number_of_reps = size(smallimgs)[2])
  smallimgs = network.layers[2].hidden_reps
  network_2 = net([size(smallimgs)[1],2],["input","pool"])
  # get_moving_vbar: present images in certain order!
  errors = learn_layer_pool!(network_2.layers[1], network_2.layers[2], get_moving_vbar, iterations)
  generatehiddenreps(network_2.layers[1], network_2.layers[2], number_of_reps = size(smallimgs)[2])

  save(string(getsavepath(),"SparsePooling/analysis/tests/pool_",dataset_pool,".jld"),
      "network", network_2, "squared_errors", errors)
end


# plotting
# ws = zeros(12*20,12*10)
# for i in 1:20
#   for j in 1:10
#     ws[(i-1)*12+1:i*12,(j-1)*12+1:j*12] = reshape(network.layers[2].w[(i-1)*10+j,:],12,12)
#   end
# end

# Testing spike trigered response equals feed-forward weights:
# str = network.layers[2].hidden_reps*smallimgs'
# str_ws = zeros(12*20,12*10)
# for i in 1:20
#   for j in 1:10
#     str_ws[(i-1)*12+1:i*12,(j-1)*12+1:j*12] = reshape(str[(i-1)*10+j,:],12,12)
#   end
# end
