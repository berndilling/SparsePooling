
using StatsBase, ProgressMeter, JLD, PyPlot
close("all")
include("./sparsepooling/sparsepooling_import.jl")

iterations = 10^4#6
in_size = 64
hidden_size = 16#32

network = net([in_size,hidden_size],["input","sparse"])
set_init_bars!(network.layers[2],hidden_size)
network.layers[2].parameters.p = 1/(16)#1/50 #2.5/48
#network.layers[2].parameters.activationfunction = "relu"

smallimgs = zeros(64,iterations)
for i in 1:iterations
 smallimgs[:,i] = get_connected_pattern()[12:19,12:19][:]
 #smallimgs[:,i] = get_connected_pattern()[25:32,25:32][:]
end

errors, ffd = learn_layer_sparse!(network.layers[1], network.layers[2], getsmallimg, iterations)
generatehiddenreps(network.layers[1], network.layers[2], smallimgs; number_of_reps = 10^4)

print("\n")
print(getsparsity(network.layers[2].hidden_reps[:]))
print("\n")
print(mean(network.layers[2].hidden_reps[:]))

# ws = zeros(8*8,8*6)
# for i in 1:8
#   for j in 1:6
#     ws[(i-1)*8+1:i*8,(j-1)*8+1:j*8] = reshape(network.layers[2].w[(i-1)*6+j,:],8,8)
#   end
# end
ws = zeros(8*4,8*4)
for i in 1:4
  for j in 1:4
    ws[(i-1)*8+1:i*8,(j-1)*8+1:j*8] = reshape(network.layers[2].w[(i-1)*4+j,:],8,8)
  end
end
figure()
title("SC receptive fields") # = feedforward weights
imshow(ws)

figure()
title("lateral inhibtion weights")
imshow(network.layers[2].v)

figure()
title("hist of hidden reps activations")
PyPlot.plt[:hist](network.layers[2].hidden_reps[:],bins = 20, histtype = "step", color = "k", log = "true")

figure()
title("hist of weight values, w:black, v:red")
PyPlot.plt[:hist](network.layers[2].w[:],bins = 10, histtype = "step", color = "k")
PyPlot.plt[:hist](network.layers[2].v[:],bins = 10, histtype = "step", color = "r")

figure()
title("thresholds")
plot(network.layers[2].t[:])
xlabel("hidden neuron")

# background = zeros(32,32)
# pattern = zeros(32,32)
# pattern = get_connected_pattern!(pattern)
# pattern_seq = get_moving_pattern(pattern, PatternParameter(pattern_duration = 10), background = get_pattern!(background))
# for i in 1:10
#   figure()
#   imshow(pattern_seq[:,:,i])
# end
