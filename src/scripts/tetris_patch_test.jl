
using StatsBase, ProgressMeter, JLD2, FileIO, PyPlot
close("all")
include("./../sparsepooling/sparsepooling_import.jl")

BLAS.set_num_threads(1)

sparse_part = false#true#true
pool_part = false#true#true
sparse_part_2 = false#true#true
pool_part_2 = false#true
sparse_part_3 = false#true
pool_part_3 = false#true

iterations_sparse = 10^1
iterations_pool = 5*10^3
iterations_sparse_2 = 10^1
iterations_pool_2 = 10^1
iterations_sparse_3 = 10^1
iterations_pool_3 = 10^1

sparse_trace_timeconstant = 1e-4

patch_size = 8
image_size = 32
in_size = image_size^2
hidden_size_sparse = 16 # per SC patch
hidden_size_pool = 2
hidden_size_sparse_2 = 18
hidden_size_pool_2 = 9
hidden_size_sparse_3 = 6*9
hidden_size_pool_3 = 6

inputfunction = getanchoredobject
dynamicfunction =  getmovingobject#getbouncingobject#getjitteredobject
network = net([in_size,hidden_size_sparse,hidden_size_pool],["input","sparse_patchy","pool_patchy"],[1,49,49])
intermediatestates = []

################################################################################

if sparse_part
  set_init_bars!(network.layers[2],hidden_size_sparse; reinit_weights = true,
    activationfunction = sigm!, one_over_tau_a = sparse_trace_timeconstant)

  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
  	[inputfunction for i in 1:network.nr_layers-1],
  	[dynamicfunction for i in 1:network.nr_layers-1];
  	LearningFromLayer = 2,
  	LearningUntilLayer = 2)

  save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer2_sparse_patchy.jld2","layer",network.layers[2])
else
  network.layers[2] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer2_sparse_patchy.jld2","layer")
  #network.layers[2] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer2_sparse_patchy.jld2","layer")
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

  set_init_bars!(network.layers[3]; updaterule = GH_SFA_subtractrace_Sanger!,
    reinit_weights = false, one_over_tau_a = 1/6, p = 1/hidden_size_pool, activationfunction = relu!)

  #learn_layer_pool!(network.layers[2],network.layers[3],n_of_moving_patterns)
  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
    [inputfunction for i in 1:network.nr_layers-1],
    [dynamicfunction for i in 1:network.nr_layers-1];
    LearningFromLayer = 3,
    LearningUntilLayer = 3)

  save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer3_pool_patchy.jld2","layer",network.layers[3])
else
  network.layers[3] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer3_pool_patchy.jld2","layer")
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

  set_init_bars!(network.layers[4],hidden_size_sparse_2; reinit_weights = true, activationfunction = relu!, one_over_tau_a = sparse_trace_timeconstant)

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

  save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer4_sparse_patchy.jld2","layer",network.layers[4])
else
  network.layers[4] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer4_sparse_patchy.jld2","layer")
end

################################################################################

addlayer!(network, hidden_size_pool_2, "pool_patchy",
  layer_pool_patchy([hidden_size_sparse_2,hidden_size_pool_2];
  n_of_pool_layer_patches = 9))
if pool_part_2
  print("train pooling part")

  set_init_bars!(network.layers[5]; reinit_weights = true, one_over_tau_a = 1/16, p = 1/hidden_size_pool_2, activationfunction = relu!)

  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool,iterations_sparse_2,iterations_pool_2],
    [inputfunction for i in 1:network.nr_layers-1],
    [dynamicfunction for i in 1:network.nr_layers-1];
    LearningFromLayer = 5,
    LearningUntilLayer = 5)
  save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer5_pool_patchy.jld2","layer",network.layers[5])
else
  network.layers[5] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer5_pool_patchy.jld2","layer")
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
  save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer6_sparse.jld2","layer",network.layers[6])
else
  network.layers[6] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer6_sparse.jld2","layer")
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
  save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer7_pool.jld2","layer",network.layers[7])
else
  network.layers[7] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer7_pool.jld2","layer")
end
