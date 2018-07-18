
using StatsBase, ProgressMeter, JLD, PyPlot
close("all")
include("./sparsepooling/sparsepooling_import.jl")

sparse_part = false
pool_part = true

iterations = 10^4#5
in_size = 64
hidden_size = 16 # per SC patch
pool_size = 4 # per SC/pool patch

n_of_moving_patterns = 10^4

network = net([in_size,hidden_size,pool_size],["input","sparse_patchy","pool_patchy"])

if sparse_part
  network = net([in_size,hidden_size,pool_size],["input","sparse_patchy","pool"])
  for i in 1:network.layers[2].parameters.n_of_sparse_layer_patches
    set_init_bars!(network.layers[2].sparse_layer_patches[i],hidden_size)
    network.layers[2].sparse_layer_patches[i].parameters.p = 1/(16)
  end
  #network.layers[2].parameters.activationfunction = "relu"

  learn_layer_sparse_patchy!(network.layers[1], network.layers[2], iterations)
  #generatehiddenreps(network.layers[1], network.layers[2], smallimgs; number_of_reps = 10^4)

  #save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_test.jld","network",network)

#else
#  network = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_test.jld","network")
end
if pool_part
  #network = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_test.jld","network")
  network.layers[3] = layer_pool_patchy([length(network.layers[2].sparse_layer_patches[1].a),pool_size])

  print("train pooling part")
  for i in 1:network.layers[3].parameters.n_of_pool_layer_patches
    set_init_bars!(network.layers[3].pool_layer_patches[i])
    network.layers[3].pool_layer_patches[i].parameters.one_over_tau_a = 1/5 # shorter pooling time constant to not pool everything
    network.layers[3].pool_layer_patches[i].parameters.activationfunction = "lin"#"relu"
    network.layers[3].pool_layer_patches[i].parameters.updatetype = "SFA"#"SFA_subtracttrace"
    network.layers[3].pool_layer_patches[i].parameters.updatetype = "Sanger"
  end
  learn_layer_pool_patchy!(network, n_of_moving_patterns)

  #save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_pool_test.jld","network",network)
end

recfields = []
for k in 1:network.layers[2].parameters.n_of_sparse_layer_patches
  ws = zeros(8*4,8*4)
  for i in 1:4
    for j in 1:4
      ws[(i-1)*8+1:i*8,(j-1)*8+1:j*8] = reshape(network.layers[2].sparse_layer_patches[k].w[(i-1)*4+j,:],8,8)
    end
  end
  push!(recfields,ws)
  #figure()
  #title("SC receptive fields") # = feedforward weights
  #imshow(ws)
end

WS = zeros(8*4*7,8*4*7)
for i in 1:7
  for j in 1:7
    WS[(i-1)*8*4+1:i*8*4,(j-1)*8*4+1:j*8*4] = recfields[(i-1)*7+j]
  end
end
figure()
imshow(WS)
title("all rec. fields of all 49 patch-SC layers")
