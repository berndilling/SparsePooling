
using StatsBase, ProgressMeter, JLD, PyPlot
close("all")
include("./sparsepooling/sparsepooling_import.jl")

sparse_part = true
pool_part = true

#iterations = 2*10^4 #7
in_size = 64
hidden_size = 16 # per SC patch
pool_size = 200

n_of_moving_patterns = 10^4 #5

network = net([in_size,hidden_size,pool_size],["input","sparse_patchy","pool"])
intermediatestates = []

if sparse_part
  for i in 1:network.layers[2].parameters.n_of_sparse_layer_patches
    set_init_bars!(network.layers[2].sparse_layer_patches[i],hidden_size)
    network.layers[2].sparse_layer_patches[i].parameters.p = 1/(16)
  end
  #network.layers[2].parameters.activationfunction = "relu"

  #learn_layer_sparse_patchy!(network.layers[1], network.layers[2], iterations)
  learn_net_layerwise!(network,intermediatestates,n_of_moving_patterns;
  	inputfunction = get_connected_pattern,
  	dynamicfunction = get_moving_pattern,
  	LearningFromLayer = 2,
  	LearningUntilLayer = 2)


  #generatehiddenreps(network.layers[1], network.layers[2], smallimgs; number_of_reps = 10^4)

  savelayer("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_layer_sparse_patchy_netlearing.jld",network.layers[2])
  #save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_test_1.jld","network",network)

else
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_layer_sparse_patchy.jld",network.layers[2])
end
if pool_part
  #network = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_test.jld","network")
  network.layers[3] = layer_pool([length(network.layers[2].a),pool_size])

  print("train pooling part")
  set_init_bars!(network.layers[3])
  network.layers[3].parameters.one_over_tau_a = 1/5 # shorter pooling time constant to not pool everything
  network.layers[3].parameters.activationfunction = lin! #relu!
  network.layers[3].parameters.updaterule = GH_SFA_Sanger! #GH_SFA_subtractrace_Sanger!

  #learn_layer_pool!(network.layers[2],network.layers[3],n_of_moving_patterns)
  learn_net_layerwise!(network,intermediatestates,n_of_moving_patterns;
  	inputfunction = get_connected_pattern,
  	dynamicfunction = get_moving_pattern,
  	LearningFromLayer = 3,
  	LearningUntilLayer = 3)

  savelayer("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_layer_pool_netlearning.jld",network.layers[3])
  #save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_pool_test.jld","network",network)
else
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_layer_pool.jld",network.layers[3])
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

figure()
plt[:hist](network.layers[3].w[:], bins = 100, normed = true)

figure()
plot(network.layers[3].w')
