
using StatsBase, ProgressMeter, JLD, PyPlot
close("all")
include("./sparsepooling/sparsepooling_import.jl")

BLAS.set_num_threads(1)

sparse_part = true#true
pool_part = false#true
sparse_part_2 = false#true

iterations_sparse = 10^4 #7
iterations_pool = 10^4
iterations_sparse_2 = 10^3#3
patch_size = 8
image_size = 32
in_size = image_size^2
hidden_size_sparse = 16 # per SC patch
hidden_size_pool = 4
hidden_size_sparse_2 = 20

network = net([in_size,hidden_size_sparse,hidden_size_pool],["input","sparse_patchy","pool_patchy"],[1,49,49])
intermediatestates = []

if sparse_part
  for sparse_layer_patch in network.layers[2].sparse_layer_patches
    set_init_bars!(sparse_layer_patch,hidden_size_sparse)
    sparse_layer_patch.parameters.p = 1/(16)
  end
  #network.layers[2].parameters.activationfunction = "relu"

  #learn_layer_sparse_patchy!(network.layers[1], network.layers[2], iterations, dynamicfunction = get_moving_pattern)
  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
  	[get_connected_pattern for i in 1:network.nr_layers-1],
  	[get_moving_pattern,get_moving_pattern];#[get_moving_pattern for i in 1:network.nr_layers-1];
  	LearningFromLayer = 2,
  	LearningUntilLayer = 2)


  #generatehiddenreps(network.layers[1], network.layers[2], smallimgs; number_of_reps = 10^4)

  #savelayer("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_layer_sparse_patchy_netlearning_new.jld",network.layers[2])
  #save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_test_1.jld","network",network)
else
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_layer_sparse_patchy_netlearning_staticmovingstatic.jld",network.layers[2])
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


if pool_part
  print("train pooling part")

  for pool_layer_patch in network.layers[3].pool_layer_patches
    set_init_bars!(pool_layer_patch)
    pool_layer_patch.parameters.one_over_tau_a = 1/5 # shorter pooling time constant to not pool everything
    pool_layer_patch.parameters.activationfunction = lin! #relu!
    pool_layer_patch.parameters.updaterule = GH_SFA_Sanger! #GH_SFA_subtractrace_Sanger!
  end


  #learn_layer_pool!(network.layers[2],network.layers[3],n_of_moving_patterns)
  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
    [get_connected_pattern for i in 1:network.nr_layers-1],
    [get_moving_pattern for i in 1:network.nr_layers-1];
    LearningFromLayer = 3,
    LearningUntilLayer = 3)

  savelayer("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_layer_pool_netlearning_new.jld",network.layers[3])
  #save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_patchy_pool_test.jld","network",network)
else
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_layer_pool_patchy_netlearning_staticmovingstatic.jld",network.layers[3])
end


figure()
plt[:hist](network.layers[3].pool_layer_patches[1].w[:], bins = 10, normed = true)

figure()
plot(network.layers[3].pool_layer_patches[1].w[:])

if sparse_part_2
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_layer_sparse_patchy_netlearning_new.jld",network.layers[2])
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/bar_layer_pool_netlearning_new.jld",network.layers[3])

  push!(network.layers, layer_sparse_patchy([hidden_size_pool*network.layers[2].parameters.n_of_sparse_layer_patches,hidden_size_sparse_2];
    n_of_sparse_layer_patches = 9, patch_size = 0, in_fan = hidden_size_pool*9, overlap = 0, image_size = 32))

  for sparse_layer_patch in network.layers[4].sparse_layer_patches
    set_init_bars!(sparse_layer_patch,hidden_size_sparse_2)
    sparse_layer_patch.parameters.p = 1/hidden_size_sparse_2
  end
  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool,iterations_sparse_2],
  	[get_connected_pattern for i in 1:network.nr_layers-1],
  	[get_moving_pattern for i in 1:network.nr_layers-1];
  	LearningFromLayer = 4,
  	LearningUntilLayer = 4)

  figure()
  plot(network.layers[4].sparse_layer_patches[1].w')
  figure()
  imshow(network.layers[4].sparse_layer_patches[1].v)
end
