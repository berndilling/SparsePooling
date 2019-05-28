using PyPlot, JLD, HDF5

close("all")
include("./../sparsepooling/sparsepooling_import.jl")

n_images = 10

path = "/User/Documents/PhD/Projects/SparsePooling/analysis/patchy/"

patch_size = 8
image_size = 32
in_size = image_size^2
hidden_size_sparse = 16 # per SC patch
hidden_size_pool = 6#2
hidden_size_sparse_2 = 18
hidden_size_pool_2 = 9
hidden_size_sparse_3 = 90
hidden_size_pool_3 = 10

inputfunction = getobject
dynamicfunction = getmovingobject

network = net([in_size,hidden_size_sparse,hidden_size_pool],["input","sparse_patchy","pool_patchy"],[1,49,49])

addlayer!(network, hidden_size_sparse_2, "sparse_patchy",
  layer_sparse_patchy([hidden_size_pool*network.layers[2].parameters.n_of_sparse_layer_patches,hidden_size_sparse_2];
  n_of_sparse_layer_patches = 9, patch_size = 0, in_fan = hidden_size_pool*9, overlap = 0, image_size = 32))
addlayer!(network, hidden_size_pool_2, "pool_patchy",
  layer_pool_patchy([hidden_size_sparse_2,hidden_size_pool_2];
  n_of_pool_layer_patches = 9))
addlayer!(network, hidden_size_sparse_3, "sparse",
  layer_sparse([9*hidden_size_pool_2,hidden_size_sparse_3]))
addlayer!(network, hidden_size_pool_3, "pool",
  layer_pool([hidden_size_sparse_3,hidden_size_pool_3]))

loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_sparse_patchy.jld",network.layers[2])
#loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_pool_patchy.jld",network.layers[3])
#loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_pool_patchy_subtracttrace.jld",network.layers[3])
loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_pool_patchy_lc.jld",network.layers[3])
loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_sparse_patchy_2.jld",network.layers[4])
loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_pool_patchy_2.jld",network.layers[5])
loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_sparse_3.jld",network.layers[6])
loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_pool_3.jld",network.layers[7])

# loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_sparse_patchy_10to6.jld",network.layers[2])
# loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_pool_patchy_10to6.jld",network.layers[3])
# loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_sparse_patchy_2_10to6.jld",network.layers[4])
# loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_pool_patchy_2_10to6.jld",network.layers[5])
# loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_sparse_3_10to6.jld",network.layers[6])
# loadlayer!("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/objects_layer_pool_3_10to6.jld",network.layers[7])

# network.layers[2] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer2_sparse_patchy.jld2","layer")
# network.layers[3] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer3_pool_patchy.jld2","layer")
# network.layers[4] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer4_sparse_patchy.jld2","layer")
# network.layers[5] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer5_pool_patchy.jld2","layer")
# network.layers[6] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer6_sparse.jld2","layer")
# network.layers[7] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/tetris_layer7_pool.jld2","layer")


plotarray = []
for i in 1:n_images
  image = inputfunction()
  dynamicimage = dynamicfunction(image)
  for j in 1:size(dynamicimage)[3]
    network.layers[1].a = dynamicimage[:,:,j][:]
    forwardprop!(network; FPUntilLayer = network.nr_layers)

    plot_temp = []
    for layer in network.layers
      append!(plot_temp,layer.a)
    end
    if i == 1 && j == 1
      plotarray = plot_temp
    else
      plotarray = hcat(plotarray,plot_temp)
    end
  end
end

figure(figsize = (10,6))
xlabel("time")
ylabel("neuron index")
plot([0,n_images*20],[32*32,32*32],"black")
plot([0,n_images*20],[16*49,16*49] .+ [32*32,32*32],"black")
plot([0,n_images*20],[6*49,6*49] .+ [16*49,16*49] .+ [32*32,32*32],"black")
plot([0,n_images*20],[9*18,9*18] .+ [6*49,6*49] .+ [16*49,16*49] .+ [32*32,32*32],"black")
plot([0,n_images*20],[9*9,9*9] .+ [9*18,9*18] .+ [6*49,6*49] .+ [16*49,16*49] .+ [32*32,32*32],"black")
plot([0,n_images*20],[90,90] .+ [9*9,9*9] .+ [9*18,9*18] .+ [6*49,6*49] .+ [16*49,16*49] .+ [32*32,32*32],"black")
imshow(plotarray,aspect="auto",origin="lower")
