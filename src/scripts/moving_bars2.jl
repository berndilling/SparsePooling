using StatsBase, ProgressMeter, JLD2, FileIO, PyPlot
close("all")
include("./../sparsepooling/sparsepooling_import.jl")

sparse_part = false

################################################################################
## Parametes

image_size = 8
hidden_size_sparse = 2*image_size
hidden_size_pool = 2
iterations_sparse = 5*10^4
iterations_pool = 10^3
sparse_trace_timeconstant = 1e-2#1e-4

inputfunction = getbar
dynamicfunctionsparse =  getstaticobject
dynamicfunctionpool = getmovingobject#getjitteredobject##

network = net([image_size^2,hidden_size_sparse,hidden_size_pool],["input","sparse","pool"],[1,1,1])

intermediatestates = []

################################################################################
## sparse part
if sparse_part
  set_init_bars!(network.layers[2],hidden_size_sparse; reinit_weights = true,
    activationfunction = sigm!, one_over_tau_a = sparse_trace_timeconstant)

  learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
    [inputfunction for i in 1:network.nr_layers-1],
    [dynamicfunctionsparse, dynamicfunctionpool];
    LearningFromLayer = 2,
    LearningUntilLayer = 2)
    save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/singlepatchtests/bars_layer1_sparse.jld2","layer",network.layers[2])
else
  network.layers[2] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/singlepatchtests/bars_layer1_sparse.jld2","layer")
end

ws = zeros(8*4,8*4)
for i in 1:4
  for j in 1:4
    ws[(i-1)*8+1:i*8,(j-1)*8+1:j*8] = reshape(network.layers[2].w[(i-1)*4+j,:],8,8)
  end
end
figure()
title("SC rec fields")
imshow(ws)

################################################################################
## pool part

set_init_bars!(network.layers[3]; updaterule = GH_SFA_subtractrace_Sanger!,
  reinit_weights = true, one_over_tau_a = 1/8, p = 1/16,#/hidden_size_pool,
  activationfunction = sigm!)

learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
  [inputfunction for i in 1:network.nr_layers-1],
  [dynamicfunctionsparse, dynamicfunctionpool];
  LearningFromLayer = 3,
  LearningUntilLayer = 3)

################################################################################
## Plotting

figure()
title("Weights of pooling layer")
for i in 1:hidden_size_pool
  plot(network.layers[3].w[i,:])
end
xlabel("presynaptic neuron")

a = find(x -> (x > 0),network.layers[3].w[1,:])
b = find(x -> (x > 0),network.layers[3].w[2,:])
wa = zeros(image_size,image_size*length(a))
wb = zeros(image_size,image_size*length(b))
# c = find(x -> (x > 0),network.layers[3].w[3,:])
# d = find(x -> (x > 0),network.layers[3].w[4,:])
# wc = zeros(image_size,image_size*length(c))
# wd = zeros(image_size,image_size*length(d))
for i in 1:length(a)
  wa[1:image_size,(i-1)*image_size+1:i*image_size] = reshape(network.layers[2].w[a[i],:],image_size,image_size)
end
for i in 1:length(b)
  wb[1:image_size,(i-1)*image_size+1:i*image_size] = reshape(network.layers[2].w[b[i],:],image_size,image_size)
end
# for i in 1:length(c)
#   wc[1:image_size,(i-1)*image_size+1:i*image_size] = reshape(network.layers[2].w[c[i],:],image_size,image_size)
# end
# for i in 1:length(d)
#   wd[1:image_size,(i-1)*image_size+1:i*image_size] = reshape(network.layers[2].w[d[i],:],image_size,image_size)
# end

figure()
imshow(wa)
axis("off")
figure()
imshow(wb)
axis("off")
# figure()
# imshow(wc)
# axis("off")
# figure()
# imshow(wd)
# axis("off")

figure()
imshow(network.layers[3].v)
