using MAT, HDF5, JLD, PyPlot

close("all")
include("./sparsepooling/sparsepooling_import.jl")


iterations = 10^6

sparse_part = false#true
pool_part = true
generate_hr = false#true

#dataset to be used
dataset_sparse = "Olshausen_white"
dataset_pool = "Olshausen_white"
labelled = false

number_of_hidden_reps = 10^4
nr_presentations_per_patch = 50
iterations_pool = 10^6 #number_of_hidden_reps*nr_presentations_per_patch

number_of_hidden_reps_pool = 1000 #just for testing pooling

hidden_size = 100
hidden_pool = 5

smallimgs_sparse, n_samples = import_unlabelled_data(dataset_sparse)
smallimgs_pool, n_samples = import_unlabelled_data(dataset_pool)
#scale data between [-1,1]
#rescaledata!(smallimgs)

#THIS MIGHT NOT EVEN BE HELPFUL!
#substract line-wise (pixel/variable-wise) mean
subtractmean!(smallimgs_sparse)
subtractmean!(smallimgs_pool)

if sparse_part
  smallimgs = smallimgs_sparse
  #Create network with two layers: ordinary sparse coding setup
  network = net([size(smallimgs)[1],hidden_size],["input","sparse"])
  # only learn thresholds to reach stable values
  network.layers[2].parameters.learningrate_v = 0.
  network.layers[2].parameters.learningrate_w = 0.
  network.layers[2].parameters.learningrate_thr = 1e-1
  errors, ffd = learn_layer_sparse!(network.layers[1], network.layers[2], getsmallimg, 10^3)
  # learn sparse layer
  network.layers[2].parameters = parameters_sparse() #back to default parameters
  network.layers[2].parameters.p = 0.05
  network.layers[2].parameters.activationfunction = "relu"
  errors, ffd = learn_layer_sparse!(network.layers[1], network.layers[2], getsmallimg, iterations)
  generatehiddenreps(network.layers[1], network.layers[2], number_of_reps = number_of_hidden_reps)

  save(string(getsavepath(),"SparsePooling/analysis/SFA/sparse_",dataset_sparse,".jld"),
      "network", network, "squared_errors", errors, "ffd", ffd)

  figure()
  title("hidden activations of the first patterns")
  ylabel("# neuron")
  xlabel("# pattern")
  imshow(network.layers[2].hidden_reps[:,1:100])
  print(getsparsity(network.layers[2].hidden_reps[:]))
  print("\n")
end

if pool_part
  network = load(string(getsavepath(),"SparsePooling/analysis/SFA/sparse_",dataset_sparse,".jld"),"network")
  #get moved patches and corresponding SC hidden reps (=smallimgs) as input for pooling:
  #movedpatches, smallimgs = generatemovingpatches(smallimgs_sparse, network.layers[1], network.layers[2],
  #nr_presentations_per_patch = 30, number_of_patches = number_of_hidden_reps)
  #same for jittered patches
  if generate_hr
    jitteredpatches, smallimgs = generatejitteredpatches(smallimgs_sparse, network.layers[1], network.layers[2],
    max_amplitude = 5, nr_presentations_per_patch = nr_presentations_per_patch, number_of_patches = number_of_hidden_reps)
    subtractmean!(smallimgs) #Otherwise first SF/PC is the mean...
    save(string(getsavepath(),"SparsePooling/analysis/SFA/SC_hidden_rep_jittered_",dataset_pool,".jld"),
      "jitteredpatches", jitteredpatches, "hidden_reps", smallimgs)
  else
    jitteredpatches = load(string(getsavepath(),"SparsePooling/analysis/SFA/SC_hidden_rep_jittered_",dataset_sparse,".jld"),"jitteredpatches")
    smallimgs = load(string(getsavepath(),"SparsePooling/analysis/SFA/SC_hidden_rep_jittered_",dataset_sparse,".jld"),"hidden_reps")
  end
  network_2 = net([size(smallimgs)[1],hidden_pool],["input","pool"])
  network_2.layers[2].parameters.learningrate = 1e-3#1e-3
  network_2.layers[2].parameters.activationfunction = "relu"#"linear"
    network_2.layers[2].parameters.updatetype = "SFA_subtracttrace" #subtract input trace (might not be necessary)
    network_2.layers[1].parameters.one_over_tau_a = 1e-6 # subract long-term mean of input
  network_2.layers[2].parameters.updaterule = "Sanger"
  errors = learn_layer_pool!(network_2.layers[1], network_2.layers[2], getsmallimg, iterations_pool)#Int(size(smallimgs)[2]))
  generatehiddenreps(network_2.layers[1], network_2.layers[2], number_of_reps = number_of_hidden_reps_pool)
  poolingrecfields = generateratetriggeredrecfields(network_2.layers[1], network_2.layers[2], number_of_reps = Int(size(smallimgs)[2]))
  complexrecfields = generatecomplexrecfields(network_2.layers[1],network_2.layers[2],jitteredpatches)

  save(string(getsavepath(),"SparsePooling/analysis/SFA/pool_",dataset_pool,".jld"),
      "network", network_2, "squared_errors", errors)
end



#plotting
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
PyPlot.plt[:hist](network.layers[2].hidden_reps[:],bins = 20, histtype = "step", color = "k", log = "true")

figure()
title("hist of weight values, w:black, v:red")
PyPlot.plt[:hist](network.layers[2].w[:],bins = 10, histtype = "step", color = "k")
PyPlot.plt[:hist](network.layers[2].v[:],bins = 10, histtype = "step", color = "r")

figure()
title("thresholds")
plot(network.layers[2].t[:])

figure()
title("weights of pooling layer")
peaks = Array{Array{Int64, 1}, 1}()
for i in 1:hidden_pool
  plot(network_2.layers[2].w[i,:])
  push!(peaks,gethighestvalues(poolingrecfields[i,:]; number = ceil(0.05*hidden_size)))
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
