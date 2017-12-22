
using JLD, PyPlot

close("all")
include("./sparsepooling/sparsepooling_import.jl")

dataset_sparse = "Olshausen_white"
dataset_pool = "Olshausen_white"
hidden_size = 100#100
hidden_pool = 5

network = load(string(getsavepath(),"SparsePooling/analysis/SFA/sparse_nh",hidden_size,"_",dataset_sparse,".jld"),"network")
network_2 = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_nh",hidden_size,"_",dataset_pool,".jld"),"network")
poolingrecfields = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_nh",hidden_size,"_",dataset_pool,".jld"),"poolingrecfields")
complexrecfields = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_nh",hidden_size,"_",dataset_pool,".jld"),"complexrecfields")

#plotting
# ws = zeros(16*40,16*25)
# for i in 1:40
#   for j in 1:25
#     ws[(i-1)*16+1:i*16,(j-1)*16+1:j*16] = reshape(network.layers[2].w[(i-1)*25+j,:],16,16)
#   end
# end
ws = zeros(16*10,16*10)
for i in 1:10
  for j in 1:10
    ws[(i-1)*16+1:i*16,(j-1)*16+1:j*16] = reshape(network.layers[2].w[(i-1)*10+j,:],16,16)
  end
end
figure()
title("SC rec. fields (feedforward weights)")
imshow(ws)

figure()
title("lateral inhibtion weights")
imshow(network.layers[2].v)

figure()
title("hist of hidden reps activations")
PyPlot.plt[:hist](network.layers[2].hidden_reps[:],bins = 100, histtype = "step", color = "k", log = "true")

figure()
title("hist of weight values, w:black, v:red")
PyPlot.plt[:hist](network.layers[2].w[:],bins = 50, histtype = "step", color = "k")
PyPlot.plt[:hist](network.layers[2].v[:],bins = 50, histtype = "step", color = "r")

figure()
title("thresholds")
plot(network.layers[2].t[:])

figure()
title("weights of pooling layer")
peaks = Array{Array{Int64, 1}, 1}()
for i in 1:hidden_pool
  plot(network_2.layers[2].w[i,:])
  push!(peaks,gethighestvalues(poolingrecfields[i,:]; number = 5)) # number = ceil(0.05*hidden_size)
end

figure()
title("hidden activations of pooling layer")
for i in 1:hidden_pool
  plot(network_2.layers[2].hidden_reps[i,:])
end

for i in 1:hidden_pool
  for j in 1:length(peaks[i])
    figure()
    imshow(reshape(network.layers[2].w[peaks[i][j],:],16,16))
  end
  figure()
end

figure()
title("pooling rec. fields (rate triggered, av. over input)")
xlabel("input neuron")
ylabel("rate triggered response")
for i in 1:hidden_pool
  plot(poolingrecfields[i,:])
end

for i in 1:hidden_pool
  figure()
  imshow(reshape(complexrecfields[i,:],16,16))
end
