
using JLD, PyPlot

close("all")
include("./sparsepooling/sparsepooling_import.jl")

dataset_sparse = "MNIST144"
dataset_pool = "MNIST144"
hidden_size = 200#250#100
hidden_pool = 5

network = load(string(getsavepath(),"SparsePooling/analysis/SFA/sparse_nh",hidden_size,"_",dataset_sparse,".jld"),"network")
network_2 = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_nh",hidden_size,"_",dataset_pool,".jld"),"network")
poolingrecfields = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_nh",hidden_size,"_",dataset_pool,".jld"),"poolingrecfields")
complexrecfields = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_nh",hidden_size,"_",dataset_pool,".jld"),"complexrecfields")


ws = zeros(12*20,12*10)
for i in 1:20
  for j in 1:10
    ws[(i-1)*12+1:i*12,(j-1)*12+1:j*12] = reshape(network.layers[2].w[(i-1)*10+j,:],12,12)
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
PyPlot.plt[:hist](network.layers[2].w[:],bins = 100, histtype = "step", color = "k")
PyPlot.plt[:hist](network.layers[2].v[:],bins = 100, histtype = "step", color = "r")

figure()
title("thresholds")
plot(network.layers[2].t[:])

figure()
title("weights of pooling layer")
peaks = Array{Array{Int64, 1}, 1}()
for i in 1:hidden_pool
  plot(network_2.layers[2].w[i,:])
  push!(peaks,gethighestvalues(poolingrecfields[i,:]; number = 4)) # number = ceil(0.05*hidden_size)
end

figure()
title("hidden activations of pooling layer")
for i in 1:hidden_pool
  plot(network_2.layers[2].hidden_reps[i,:])
end

for i in 1:hidden_pool
  for j in 1:length(peaks[i])
    figure()
    imshow(reshape(network.layers[2].w[peaks[i][j],:],12,12))
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
  imshow(reshape(complexrecfields[i,:],12,12))
end
