
using StatsBase, ProgressMeter, JLD, PyPlot
close("all")
include("./sparsepooling/sparsepooling_import.jl")

BLAS.set_num_threads(1)

sparse_part = false#true#true
pool_part = false#true#true
sparse_part_2 = true#true#true
pool_part_2 = true#true
sparse_part_3 = false#true
pool_part_3 = false#true

iterations_sparse = 10^3
iterations_pool = 10^3
iterations_sparse_2 = 10^3
iterations_pool_2 = 10^3
iterations_sparse_3 = 10^1
iterations_pool_3 = 10^1

patch_size = 8
image_size = 32
in_size = image_size^2
hidden_size_sparse = 16 # per SC patch
hidden_size_pool = 4
hidden_size_sparse_2 = 18
hidden_size_pool_2 = 9
hidden_size_sparse_3 = 6*9
hidden_size_pool_3 = 6

inputfunction = getanchoredobject
dynamicfunction = getbouncingobject
network = net([in_size,hidden_size_sparse,hidden_size_pool],["input","sparse_patchy","pool_patchy"],[1,49,49])
intermediatestates = []

################################################################################

if sparse_part
  set_init_bars!(network.layers[2],hidden_size_sparse; reinit_weights = true, activationfunction = pwl!, one_over_tau_a = 5e-3)

  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
  	[inputfunction for i in 1:network.nr_layers-1],
  	[dynamicfunction for i in 1:network.nr_layers-1];
  	LearningFromLayer = 2,
  	LearningUntilLayer = 2)

  savelayer("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer1_sparse_patchy.jld",network.layers[2])
else
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer1_sparse_patchy.jld",network.layers[2])
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

################################################################################

if pool_part
  print("train pooling part")
  if !sparse_part
    set_init_bars!(network.layers[2],hidden_size_sparse; reinit_weights = true)
    loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer1_sparse_patchy.jld",network.layers[2])
  end

  set_init_bars!(network.layers[3]; reinit_weights = true, one_over_tau_a = 1/8, p = 1/2, activationfunction = relu!)


  #learn_layer_pool!(network.layers[2],network.layers[3],n_of_moving_patterns)
  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
    [inputfunction for i in 1:network.nr_layers-1],
    [dynamicfunction for i in 1:network.nr_layers-1];
    LearningFromLayer = 3,
    LearningUntilLayer = 3)

  savelayer("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer2_pool_patchy.jld",network.layers[3])
else
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer2_pool_patchy.jld",network.layers[3])
end


figure()
plt[:hist](network.layers[3].pool_layer_patches[1].w[:], bins = 10, normed = true)

figure()
plot(network.layers[3].pool_layer_patches[1].w[:])

################################################################################

addlayer!(network, hidden_size_sparse_2, "sparse_patchy",
  layer_sparse_patchy([hidden_size_pool*network.layers[2].parameters.n_of_sparse_layer_patches,hidden_size_sparse_2];
  n_of_sparse_layer_patches = 9, patch_size = 0, in_fan = hidden_size_pool*9, overlap = 0, image_size = 32))
if sparse_part_2
  if !sparse_part
    set_init_bars!(network.layers[2],hidden_size_sparse; reinit_weights = true)
    loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer1_sparse_patchy.jld",network.layers[2])
  end
  if !pool_part
    set_init_bars!(network.layers[3]; reinit_weights = true, one_over_tau_a = 1/8, p = 1/2, activationfunction = relu!)
    loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer2_pool_patchy.jld",network.layers[3])
  end

  set_init_bars!(network.layers[4],hidden_size_sparse_2; reinit_weights = true, activationfunction = relu!, one_over_tau_a = 5e-3)

  figure()
  plot(network.layers[4].sparse_layer_patches[1].w')
  figure()
  imshow(network.layers[4].sparse_layer_patches[1].v)

  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool,iterations_sparse_2],
  	[inputfunction for i in 1:network.nr_layers-1],
  	[dynamicfunction for i in 1:network.nr_layers-1];
  	LearningFromLayer = 4,
  	LearningUntilLayer = 4)

  figure()
  plot(network.layers[4].sparse_layer_patches[1].w')
  figure()
  imshow(network.layers[4].sparse_layer_patches[1].v)
  savelayer("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer3_sparse_patchy.jld",network.layers[4])
else
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer3_sparse_patchy.jld",network.layers[4])
end

################################################################################

addlayer!(network, hidden_size_pool_2, "pool_patchy",
  layer_pool_patchy([hidden_size_sparse_2,hidden_size_pool_2];
  n_of_pool_layer_patches = 9))
if pool_part_2
  if !sparse_part
    set_init_bars!(network.layers[2],hidden_size_sparse; reinit_weights = true)
    loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer1_sparse_patchy.jld",network.layers[2])
  end
  if !pool_part
    set_init_bars!(network.layers[3]; reinit_weights = true, one_over_tau_a = 1/8, p = 1/2, activationfunction = relu!)
    loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer2_pool_patchy.jld",network.layers[3])
  end
  if !sparse_part_2
    set_init_bars!(network.layers[4],hidden_size_sparse_2; reinit_weights = true, activationfunction = relu!, one_over_tau_a = 5e-3)
    loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer3_sparse_patchy.jld",network.layers[4])
  end
  print("train pooling part")

  set_init_bars!(network.layers[5]; reinit_weights = true, one_over_tau_a = 1/16, p = 1/hidden_size_pool_2, activationfunction = relu!)

  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool,iterations_sparse_2,iterations_pool_2],
    [inputfunction for i in 1:network.nr_layers-1],
    [dynamicfunction for i in 1:network.nr_layers-1];
    LearningFromLayer = 5,
    LearningUntilLayer = 5)
  savelayer("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer4_pool_patchy.jld",network.layers[5])
else
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer4_pool_patchy.jld",network.layers[5])
end

################################################################################

addlayer!(network, hidden_size_sparse_3, "sparse",
  layer_sparse([9*hidden_size_pool_2,hidden_size_sparse_3]))
if sparse_part_3
  print("train sparse part")
  set_init_bars!(network.layers[6],hidden_size_sparse_3)
  network.layers[6].parameters.p = 1/hidden_size_sparse_3

  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool,
      iterations_sparse_2,iterations_pool_2,iterations_sparse_3],
    [inputfunction for i in 1:network.nr_layers-1],
    [dynamicfunction for i in 1:network.nr_layers-1];
    LearningFromLayer = 6,
    LearningUntilLayer = 6)
  savelayer("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer5_sparse.jld",network.layers[6])
else
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer5_sparse.jld",network.layers[6])
end

################################################################################

addlayer!(network, hidden_size_pool_3, "pool",
  layer_pool([hidden_size_sparse_3,hidden_size_pool_3]))
if pool_part_3
  print("train pooling part")
  set_init_bars!(network.layers[3]; one_over_tau_a = 1/32, p = 1/hidden_size_pool_3, activationfunction = relu!)

  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool,
      iterations_sparse_2,iterations_pool_2,iterations_sparse_3,iterations_pool_3],
    [inputfunction for i in 1:network.nr_layers-1],
    [dynamicfunction for i in 1:network.nr_layers-1];
    LearningFromLayer = 7,
    LearningUntilLayer = 7)
  savelayer("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer6_pool.jld",network.layers[7])
else
  loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer6_pool.jld",network.layers[7])
end
