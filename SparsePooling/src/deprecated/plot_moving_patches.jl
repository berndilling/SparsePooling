
using JLD, PyPlot

#close("all")
include("./sparsepooling/sparsepooling_import.jl")

dataset_sparse = "Olshausen_white"
dataset_pool = "Olshausen_white"
hidden_size = 400#250#100
hidden_pool = 8#8#4#5

# network = load(string(getsavepath(),"SparsePooling/analysis/SFA/sparse_nh",hidden_size,"_",dataset_sparse,".jld"),"network")
# network_2 = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_nh",hidden_size,"_",dataset_pool,".jld"),"network")
# poolingrecfields = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_nh",hidden_size,"_",dataset_pool,".jld"),"poolingrecfields")
# complexrecfields = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_nh",hidden_size,"_",dataset_pool,".jld"),"complexrecfields")
network = load(string(getsavepath(),"SparsePooling/analysis/SFA/sparse_boost_nh",hidden_size,"_",dataset_sparse,".jld"),"network")
network_2 = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_boost_nh",hidden_size,"_",dataset_pool,".jld"),"network")
poolingrecfields = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_boost_nh",hidden_size,"_",dataset_pool,".jld"),"poolingrecfields")
complexrecfields = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_boost_nh",hidden_size,"_",dataset_pool,".jld"),"complexrecfields")

# network = load(string(getsavepath(),"SparsePooling/analysis/SFA/sparse_relu_nh",hidden_size,"_",dataset_sparse,".jld"),"network")
# network_2 = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_relu_nh",hidden_size,"_",dataset_pool,".jld"),"network")
# poolingrecfields = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_relu_nh",hidden_size,"_",dataset_pool,".jld"),"poolingrecfields")
# complexrecfields = load(string(getsavepath(),"SparsePooling/analysis/SFA/pool_relu_nh",hidden_size,"_",dataset_pool,".jld"),"complexrecfields")
print("\n")
print(getsparsity(network.layers[2].hidden_reps[:]))
print("\n")
print(getsparsity(network_2.layers[2].hidden_reps[:]))

ws = zeros(16*20,16*20)
for i in 1:20
  for j in 1:20
    ws[(i-1)*16+1:i*16,(j-1)*16+1:j*16] = reshape(network.layers[2].w[(i-1)*20+j,:],16,16)
  end
end
# ws = zeros(16*10,16*10)
# for i in 1:10
#   for j in 1:10
#     ws[(i-1)*16+1:i*16,(j-1)*16+1:j*16] = reshape(network.layers[2].w[(i-1)*10+j,:],16,16)
#   end
# end

figure()
title(string("All ",hidden_size, " SC receptive fields"))
imshow(ws)
axis("off")
savefig("/Users/Bernd/Documents/PhD/Presentations/LabMeeting_February18/patches_SC_boost_rec_fields.pdf")

figure()
title("lateral inhibtion weights")
imshow(network.layers[2].v)
axis("off")
savefig("/Users/Bernd/Documents/PhD/Presentations/LabMeeting_February18/patches_SC_boost_lateralinhibitionweights.pdf")

figure()
title("SC hidden activations")
xlabel("rate")
ylabel("counts")
PyPlot.plt[:hist](network.layers[2].hidden_reps[:],bins = 100, histtype = "step", color = "k", log = "true")
savefig("/Users/Bernd/Documents/PhD/Presentations/LabMeeting_February18/patches_SC_boost_hiddenactivations.pdf")

figure()
title("hist of weight values, w:black, v:red")
PyPlot.plt[:hist](network.layers[2].w[:],bins = 100, histtype = "step", color = "k")
PyPlot.plt[:hist](network.layers[2].v[:],bins = 100, histtype = "step", color = "r")

figure()
title("thresholds")
plot(network.layers[2].t[:])

figure()
title("weights of pooling layer")
xlabel("presynaptic neuron")
peaks = Array{Array{Int64, 1}, 1}()
for i in 1:hidden_pool
  plot(network_2.layers[2].w[i,:])
  push!(peaks,gethighestvalues(poolingrecfields[i,:]; number = 10)) # number = ceil(0.05*hidden_size)
end
savefig("/Users/Bernd/Documents/PhD/Presentations/LabMeeting_February18/patches_boost_poolingweights.pdf")

figure()
title("hidden activations of pooling layer")
for i in 1:hidden_pool
  plot(network_2.layers[2].hidden_reps[i,:])
end
xlabel("timestep (20 steps per pattern)")
ylabel("rate")
savefig("/Users/Bernd/Documents/PhD/Presentations/LabMeeting_February18/patches_boost_hiddenactivations.pdf")

ws = zeros(16*hidden_pool,16*10)
for i in 1:hidden_pool
  for j in 1:10
    ws[(i-1)*16+1:i*16,(j-1)*16+1:j*16] = reshape(network.layers[2].w[peaks[i][j],:],16,16)
  end
end
figure()
imshow(ws)
axis("off")
ylabel("pooling neuron")
xlabel("pooled SC fields")
savefig("/Users/Bernd/Documents/PhD/Presentations/LabMeeting_February18/patches_boost_pooledSCfields.pdf")

figure()
title("pooling receptive fields")
xlabel("presynaptic neuron")
ylabel("rate triggered response")
for i in 1:hidden_pool
  plot(poolingrecfields[i,:])
end
savefig("/Users/Bernd/Documents/PhD/Presentations/LabMeeting_February18/patches_boost_poolingrecfields.pdf")

figure()
ws = zeros(16*2,16*4)
for i in 1:2
  for j in 1:4
    ws[(i-1)*16+1:i*16,(j-1)*16+1:j*16] = reshape(complexrecfields[(i-1)*4+j,:],16,16)
  end
end
imshow(ws)
title("complex fields")
axis("off")
savefig("/Users/Bernd/Documents/PhD/Presentations/LabMeeting_February18/patches_boost_complexfields.pdf")
