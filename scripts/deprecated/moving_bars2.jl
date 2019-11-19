
# This script reproduces the moving bars script in the old framework
# Works both with:
# 1. Generalized Hebbian rule (GH_SFA_subtractrace_Sanger)
# OR
# 2. Variant of Földiak (SC) rule (change keyword lc_forward = true in forwardprop and parameterupdate)

using StatsBase, ProgressMeter, JLD2, FileIO, PyPlot
close("all")
include("./../sparsepooling/sparsepooling_import.jl")

sparse_part = true

################################################################################
## Parametes

image_size = 4
hidden_size_sparse = 2*image_size#2*image_size
hidden_size_pool = 2
iterations_sparse = 10^5#5
iterations_pool = 10^4#2
# actually: one over timeconstant
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
    #save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/singlepatchtests/bars_layer2_sparse_2tau_sigma_s.jld2","layer",network.layers[2])
else
  network.layers[2] = load("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/singlepatchtests/bars_layer2_sparse_2tau.jld2","layer")
end

ws = zeros(4*4,2*4)
for i in 1:4
  for j in 1:2
    ws[(i-1)*4+1:i*4,(j-1)*4+1:j*4] = reshape(network.layers[2].w[(i-1)*2+j,:],4,4)
  end
end
# ws = zeros(4*4,5*4)
# for i in 1:4
#   for j in 1:5
#     ws[(i-1)*4+1:i*4,(j-1)*4+1:j*4] = reshape(network.layers[2].w[(i-1)*5+j,:],4,4)
#   end
# end
figure()
title("SC rec fields")
imshow(ws)

################################################################################
## pool part

set_init_bars!(network.layers[3]; updaterule = GH_SFA_subtractrace_Sanger!,
  reinit_weights = true, one_over_tau_a = 1/4, p = 1/2,# one_over_tau_a = 1/4, p = 1/5 or 1/8
  activationfunction = sigm_s!) #sigm!

learn_net_layerwise!(network,intermediatestates,[iterations_sparse,iterations_pool],
  [inputfunction for i in 1:network.nr_layers-1],
  [dynamicfunctionsparse, dynamicfunctionpool];
  LearningFromLayer = 3,
  LearningUntilLayer = 3)

#save("/Users/Bernd/Documents/PhD/Projects/SparsePooling/analysis/patchy/singlepatchtests/bars_layer3_pool_2tau_sigma_s.jld2","layer",network.layers[3])
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
#c = find(x -> (x > 0),network.layers[3].w[3,:])
wa = zeros(image_size,image_size*length(a))
wb = zeros(image_size,image_size*length(b))
#wc = zeros(image_size,image_size*length(c))
for i in 1:length(a)
  wa[1:image_size,(i-1)*image_size+1:i*image_size] = reshape(network.layers[2].w[a[i],:],image_size,image_size)
end
for i in 1:length(b)
  wb[1:image_size,(i-1)*image_size+1:i*image_size] = reshape(network.layers[2].w[b[i],:],image_size,image_size)
end
# for i in 1:length(c)
#   wc[1:image_size,(i-1)*image_size+1:i*image_size] = reshape(network.layers[2].w[c[i],:],image_size,image_size)
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

figure()
imshow(network.layers[3].v)
