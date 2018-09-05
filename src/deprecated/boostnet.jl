
#construct larger network by rotations of receptive fields

using JLD, PyPlot

boostfactor = 4
number_of_hidden_reps = 10^4

include("/Users/Bernd/Documents/PhD/Projects/SparsePooling/src/sparsepooling/sparsepooling_import.jl")
dataset_sparse = "Olshausen_white"
hidden_size = 100
network = load(string(getsavepath(),"SparsePooling/analysis/SFA/sparse_nh",hidden_size,"_",dataset_sparse,".jld"),"network")

network_boost = net([256,boostfactor*hidden_size],["input","sparse"])
network_boost.layers[2].parameters.activationfunction = "relu"

for i in 1:boostfactor
  for j in 1:hidden_size
    network_boost.layers[2].w[(i-1)*hidden_size+j,:] = rotl90(reshape(deepcopy(network.layers[2].w[j,:]),16,16),(i-1))[:]
  end
  network_boost.layers[2].v[(i-1)*hidden_size+1:i*hidden_size,(i-1)*hidden_size+1:i*hidden_size] = deepcopy(network.layers[2].v)
  network_boost.layers[2].t[(i-1)*hidden_size+1:i*hidden_size] = deepcopy(network.layers[2].t)
end

figure()
imshow(network_boost.layers[2].v)

ws = zeros(16*20,16*20)
for i in 1:20
  for j in 1:20
    ws[(i-1)*16+1:i*16,(j-1)*16+1:j*16] = reshape(network_boost.layers[2].w[(i-1)*20+j,:],16,16)
  end
end
figure()
imshow(ws)

smallimgs, n_samples = import_unlabelled_data(dataset_sparse)
subtractmean!(smallimgs)
generatehiddenreps(network_boost.layers[1], network_boost.layers[2], number_of_reps = number_of_hidden_reps)

save(string(getsavepath(),"SparsePooling/analysis/SFA/sparse_boost_nh",boostfactor*hidden_size,"_",dataset_sparse,".jld"),
    "network", network_boost)
