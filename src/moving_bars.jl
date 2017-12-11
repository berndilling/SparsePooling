using MAT, HDF5, JLD, PyPlot

close("all")
include("./sparsepooling/sparsepooling_import.jl")

sparse_part = false#true
pool_part = true

#dataset to be used
dataset_sparse = "bars"#"bars_superimposed"
dataset_pool = "bars"
labelled = false

hidden_size = 24
hidden_pool = 2

iterations = 10^5

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
  network.layers[2].p = 1/24
  #set initialization appropriate to bar-data
  set_init_bars!(network.layers[2],hidden_size)
  # only learn thresholds to reach stable values
  network.layers[2].parameters.learningrate_v = 0.
  network.layers[2].parameters.learningrate_w = 0.
  network.layers[2].parameters.learningrate_thr = 1e-1
  errors, ffd = learn_layer_sparse!(network.layers[1], network.layers[2], getsmallimg, 10^3)
  # learn sparse layer
  #network.layers[2].parameters = parameters_sparse() #back to default parameters
  network.layers[2].parameters.learningrate_v = 1e-1
  network.layers[2].parameters.learningrate_w = 2e-2
  network.layers[2].parameters.learningrate_thr = 2e-2
  errors, ffd = learn_layer_sparse!(network.layers[1], network.layers[2], getsmallimg, iterations)
  generatehiddenreps(network.layers[1], network.layers[2], number_of_reps = size(smallimgs)[2])

  save(string(getsavepath(),"SparsePooling/analysis/subtracttrace/sparse_",dataset_sparse,".jld"),
      "network", network, "squared_errors", errors, "ffd", ffd)

  figure()
  title("hidden activations of the first patterns")
  imshow(network.layers[2].hidden_reps[:,1:24])
  print(getsparsity(network.layers[2].hidden_reps[:]))
  print("\n")
  print(mean(network.layers[2].hidden_reps[:]))
  print("\n")
  print(network.layers[2].hidden_reps*network.layers[2].hidden_reps'./size(smallimgs)[2])
end

if pool_part
  network = load(string(getsavepath(),"SparsePooling/analysis/subtracttrace/sparse_",dataset_sparse,".jld"),"network")
  smallimgs = smallimgs_pool
  generatehiddenreps(network.layers[1], network.layers[2], number_of_reps = size(smallimgs)[2])
  smallimgs = deepcopy(network.layers[2].hidden_reps)
  subtractmean!(smallimgs) #Otherwise first SF/PC is the mean...
  network_2 = net([size(smallimgs)[1],hidden_pool],["input","pool"])
  network_2.layers[2].parameters.nonlinearity = "linear" #relu works nice but no idea why!
  network_2.layers[2].parameters.updatetype = "SFA"#"SFA_subtracttrace"
  network_2.layers[2].parameters.updaterule = "Sanger"
  network_2.layers[2].parameters.learningrate = 1e-2
  network_2.layers[2].parameters.one_over_tau_a = 1e-1
  # get_moving_vbar: present images in certain order!
  errors = learn_layer_pool!(network_2.layers[1], network_2.layers[2], get_jittered_bar, iterations)#get_moving_vbar #get_moving_vbar
  generatehiddenreps(network_2.layers[1], network_2.layers[2], number_of_reps = size(smallimgs)[2])

  save(string(getsavepath(),"SparsePooling/analysis/subtracttrace/pool_",dataset_pool,".jld"),
      "network", network_2, "squared_errors", errors)
end



#plotting
ws = zeros(12*4,12*6)
for i in 1:4
  for j in 1:6
    ws[(i-1)*12+1:i*12,(j-1)*12+1:j*12] = reshape(network.layers[2].w[(i-1)*4+j,:],12,12)
  end
end
figure()
title("SC rec. fields (feedforward weights)")
imshow(ws)

figure()
title("lateral inhibtion weights")
imshow(network.layers[2].v)

figure()
title("hidden activations of the 24 pooling patterns")
ylabel("# neuron")
xlabel("# pattern")
imshow(network.layers[2].hidden_reps)

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
for i in 1:hidden_pool
  plot(network_2.layers[2].w[i,:])
end

figure()
title("hidden activations of pooling layer")
for i in 1:hidden_pool
  plot(network_2.layers[2].hidden_reps[i,:])
end

# Testing spike trigered response equals feed-forward weights:
# str = network.layers[2].hidden_reps*smallimgs'
# str_ws = zeros(12*4,12*6)
# for i in 1:4
#   for j in 1:6
#     str_ws[(i-1)*12+1:i*12,(j-1)*12+1:j*12] = reshape(str[(i-1)*4+j,:],12,12)
#   end
# end
# figure()
# imshow(str_ws)
