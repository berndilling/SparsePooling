
using StatsBase, ProgressMeter, JLD2, FileIO, PyPlot
close("all")
#@everywhere
include("./../sparsepooling/sparsepooling_import.jl")

#BLAS.set_num_threads(1)

sparse_part = false#true
pool_part = true#true#true
sparse_part_2 = false#true#true#true
pool_part_2 = false#true
sparse_part_3 = false#true
pool_part_3 = false#true

iterations_sparse = 10^4
iterations_pool = 10^4
iterations_sparse_2 = 10^4
iterations_pool_2 = 10^1
iterations_sparse_3 = 10^1
iterations_pool_3 = 10^1

# ATTENTION: this is overwritten when loading first layer!
sparse_trace_timeconstant = 1e-2#1e-4
image_size = 32#32
in_size = image_size^2

patch_size = 2
overlap = 0
hidden_size_sparse = 10#8 # per SC patch
hidden_size_pool = 4#2
hidden_size_sparse_2 = 40#8
# hidden_size_pool_2 = 9
# hidden_size_sparse_3 = 6*9
# hidden_size_pool_3 = 6

inputfunction = getanchoredobject#getbar#gettriangle#  getimage?
dynamicfunction = getmovingobject#getbouncingobject#getjitteredobject#getstaticobject#  getmovingimage?


network = net([in_size],["input"])
addlayer!(network, hidden_size_sparse, "sparse_patchy",
  layer_sparse_patchy([in_size,hidden_size_sparse];
    patch_size = patch_size, in_fan = patch_size^2, overlap = overlap, image_size = image_size))
addlayer!(network, hidden_size_pool, "pool_patchy",
  layer_pool_patchy([hidden_size_sparse,hidden_size_pool]; patch_size = patch_size, overlap = overlap, image_size = image_size))

intermediatestates = []

################################################################################

if sparse_part
  set_init_bars!(network.layers[2],hidden_size_sparse; reinit_weights = true,
    activationfunction = sigm!, one_over_tau_a = sparse_trace_timeconstant,
    p = 1. / hidden_size_sparse)

  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
  	[inputfunction for i in 1:network.nr_layers-1],
  	[dynamicfunction for i in 1:network.nr_layers-1];
  	LearningFromLayer = 2,
  	LearningUntilLayer = 2)

  save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/julia1.0/tetris_no_overlap_layer2_sparse_patchy.jld2","layer",network.layers[2])
else
  network.layers[2] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/julia1.0/tetris_no_overlap_layer2_sparse_patchy.jld2","layer")
end

#loadsharedweights!(network.layers[2],"/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/singlepatchtests/bars_layer2_sparse_2tau_sigma_s.jld2")

recfields = []
split1 = 2 #4
split2 = Int(hidden_size_sparse/split1)
for k in 1:network.layers[2].parameters.n_of_sparse_layer_patches
  ws = zeros(patch_size*split1,patch_size*split2)
  for i in 1:split1
    for j in 1:split2
      ws[(i-1)*patch_size+1:i*patch_size,(j-1)*patch_size+1:j*patch_size] =
        reshape(network.layers[2].sparse_layer_patches[k].w[(i-1)*split2+j,:],patch_size,patch_size)
    end
  end
  push!(recfields,ws)
end
figure()
n_of_patches_sqrt = Int(image_size/patch_size)
imshow(recfields[Int(n_of_patches_sqrt^2/2)])
WS = zeros(n_of_patches_sqrt*patch_size*split1,n_of_patches_sqrt*patch_size*split2)
for i in 1:n_of_patches_sqrt
  for j in 1:n_of_patches_sqrt
    WS[(i-1)*patch_size*split1+1:i*patch_size*split1,(j-1)*patch_size*split2+1:j*patch_size*split2] = recfields[(i-1)*n_of_patches_sqrt+j]
  end
end
figure()
imshow(WS)
title(string("all rec. fields of all ",n_of_patches_sqrt^2," patch-SC layers"))

################################################################################

if pool_part
  print("train pooling part")

  set_init_bars!(network.layers[3]; updaterule = GH_SFA_subtractrace_Sanger!,
    reinit_weights = true, one_over_tau_a = 1/4, p = 1/(2*hidden_size_pool),# p = 1/(2*hidden_size_pool),#one_over_tau_a = 1/4, p = 1/5
    activationfunction = sigm!) # sigm_s!

    figure()
    plt[:hist](network.layers[3].pool_layer_patches[Int(n_of_patches_sqrt^2/2)].w[:], bins = 10, normed = true)

    figure()
    temp = copy(network.layers[3].pool_layer_patches[Int(n_of_patches_sqrt^2/2)].w')
    plot(temp)

  #learn_layer_pool!(network.layers[2],network.layers[3],n_of_moving_patterns)
  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
    [inputfunction for i in 1:network.nr_layers-1],
    [dynamicfunction for i in 1:network.nr_layers-1];
    LearningFromLayer = 3,
    LearningUntilLayer = 3)

  save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/julia1.0/tetris_no_overlap_layer3_pool_patchy.jld2","layer",network.layers[3])
else
  network.layers[3] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/julia1.0/tetris_no_overlap_layer3_pool_patchy.jld2","layer")
end

#loadsharedweights!(network.layers[3],"/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/singlepatchtests/bars_layer3_pool_2tau_sigma_s.jld2")

figure()
plt[:hist](network.layers[3].pool_layer_patches[Int(n_of_patches_sqrt^2/2)].w[:], bins = 10, normed = true)

figure()
temp = copy(network.layers[3].pool_layer_patches[Int(n_of_patches_sqrt^2/2)].w')
plot(temp)

################################################################################

addlayer!(network, hidden_size_sparse_2, "sparse_patchy",
  layer_sparse_patchy([hidden_size_pool*network.layers[2].parameters.n_of_sparse_layer_patches,hidden_size_sparse_2];
  patch_size = 2*patch_size, in_fan = hidden_size_pool*4, overlap = 0, image_size = 32))
if sparse_part_2

  set_init_bars!(network.layers[4],hidden_size_sparse_2; reinit_weights = true,
        activationfunction = sigm_m!,#sigm_s!
        one_over_tau_a = sparse_trace_timeconstant,
        p = 2 ./ hidden_size_sparse_2)

  # figure()
  # plot(network.layers[4].sparse_layer_patches[1].w')
  # figure()
  # imshow(network.layers[4].sparse_layer_patches[1].v)

  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool,iterations_sparse_2],
  	[inputfunction for i in 1:network.nr_layers-1],
  	[dynamicfunction for i in 1:network.nr_layers-1];
  	LearningFromLayer = 4,
  	LearningUntilLayer = 4)

  figure()
  temp = copy(network.layers[4].sparse_layer_patches[1].w')
  plot(temp)
  figure()
  imshow(network.layers[4].sparse_layer_patches[1].v)

  figure()
  imshow(network.layers[4].sparse_layer_patches[1].w)
  save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/julia1.0/tetris_no_overlap_layer4_sparse_patchy.jld2","layer",network.layers[4])
else
  #network.layers[4] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_no_overlap_layer4_sparse_patchy.jld2","layer")
end
#
# ################################################################################
#
# addlayer!(network, hidden_size_pool_2, "pool_patchy",
#   layer_pool_patchy([hidden_size_sparse_2,hidden_size_pool_2];
#   n_of_pool_layer_patches = 9))
# if pool_part_2
#   print("train pooling part")
#
#   set_init_bars!(network.layers[5]; reinit_weights = true, one_over_tau_a = 1/16, p = 1/hidden_size_pool_2, activationfunction = relu!)
#
#   learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool,iterations_sparse_2,iterations_pool_2],
#     [inputfunction for i in 1:network.nr_layers-1],
#     [dynamicfunction for i in 1:network.nr_layers-1];
#     LearningFromLayer = 5,
#     LearningUntilLayer = 5)
#   save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_no_overlap_layer5_pool_patchy.jld2","layer",network.layers[5])
# else
#   network.layers[5] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_no_overlap_layer5_pool_patchy.jld2","layer")
# end
#
# ################################################################################
#
# addlayer!(network, hidden_size_sparse_3, "sparse",
#   layer_sparse([9*hidden_size_pool_2,hidden_size_sparse_3]))
# if sparse_part_3
#   print("train sparse part")
#   set_init_bars!(network.layers[6],hidden_size_sparse_3)
#   network.layers[6].parameters.p = 1/hidden_size_sparse_3
#
#   learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool,
#       iterations_sparse_2,iterations_pool_2,iterations_sparse_3],
#     [inputfunction for i in 1:network.nr_layers-1],
#     [dynamicfunction for i in 1:network.nr_layers-1];
#     LearningFromLayer = 6,
#     LearningUntilLayer = 6)
#   save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_no_overlap_layer6_sparse.jld2","layer",network.layers[6])
# else
#   network.layers[6] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_no_overlap_layer6_sparse.jld2","layer")
# end
#
# ################################################################################
#
# addlayer!(network, hidden_size_pool_3, "pool",
#   layer_pool([hidden_size_sparse_3,hidden_size_pool_3]))
# if pool_part_3
#   print("train pooling part")
#   set_init_bars!(network.layers[3]; one_over_tau_a = 1/32, p = 1/hidden_size_pool_3, activationfunction = relu!)
#
#   learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool,
#       iterations_sparse_2,iterations_pool_2,iterations_sparse_3,iterations_pool_3],
#     [inputfunction for i in 1:network.nr_layers-1],
#     [dynamicfunction for i in 1:network.nr_layers-1];
#     LearningFromLayer = 7,
#     LearningUntilLayer = 7)
#   save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_no_overlap_layer7_pool.jld2","layer",network.layers[7])
# else
#   network.layers[7] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_no_overlap_layer7_pool.jld2","layer")
# end
