#####################################################
# input helpers

@inline function parallelrange(i, N, nthreads, number_of_jobs)
	if i == nthreads
			range = (i-1)*N + 1:number_of_jobs
	else
			range = (i-1)*N + 1:i * N
	end
end

@inline function getsparsity(input::Array{Float64, 1}; thr = 0.)
	length(findall(x -> (x <= thr),input))/length(input)
end

@inline function getsmallimg()
    global patternindex = rand(1:size(smallimgs)[2])
    smallimgs[:, patternindex]
end
@inline function getlabel()
    [labels[patternindex] == i for i in 0:9]
end

@inline function getsmallimg(iteration) #select images in fixed order
		if iteration > size(smallimgs)[2] iteration = (iteration % size(smallimgs)[2])+1 end
    smallimgs[:, iteration]
end

@inline function getsavepath()
	if is_apple()
		path = "/Users/Bernd/Documents/PhD/Projects/SparsePooling/"
	elseif is_linux()
		path = "/home/illing/"
	end
end

###########################################################################
# Classifier helpers
# target must be one-hot coded

function _seterror_mse!(net, target)
	net.e[end] = target - net.a[end]
end

function _loss_mse(net, target)
	return norm(net.e[end])
end

function _softmax(input)
	input = deepcopy(input)
	exps = exp.(input .- maximum(input))
  return exps / sum(exps)
end

#target must be one-hot
function _seterror_crossentropysoftmax!(net, target)
	probs = _softmax(net.a[end])
	net.e[end] = target - probs
end

function _loss_crossentropy(net, target)
	probs = _softmax(net.a[end])
	return -target'*log.(probs)
end

function geterrors!(net, imgs, labels; getwrongindices = false, noftest = size(imgs)[2])
	print("calculate classification errors...")
	error = 0
	if getwrongindices
		wrongindices = []
		@showprogress for i in 1:noftest
			net.layers[1].a = imgs[:,i]
			forwardprop!(net)
			if findmax(net.layers[end].a[end])[2] != Int(labels[i] + 1)
				error += 1
				push!(wrongindices,i)
			end
		end
		return error/noftest, wrongindices
	else
		@showprogress for i in 1:noftest
			net.layers[1].a = imgs[:,i]
			forwardprop!(net)
			error += findmax(net.layers[end].a[end])[2] != Int(labels[i] + 1)
		end
		error/noftest
	end
end

@inline function generatehiddenreps!(network::net, imgs;
					ind = size(imgs)[2], normalize = true,
					subtractmean = false)
	reps = zeros(length(network.layers[network.nr_layers].a),ind)
	@info("calculate hidden reps")
	@showprogress for i in 1:ind
	    network.layers[1].a = imgs[:,i]
	    forwardprop!(network, FPUntilLayer = network.nr_layers)
	    reps[:,i] = deepcopy(network.layers[network.nr_layers].a)
	end
	if normalize
		maximum(reps) != 0 && (reps ./= maximum(reps))
	end
	if subtractmean
		subtractmean!(reps)
	end
	return reps
end

function traintopendclassifier!(network, imgs, imgstest, labels, labelstest;
			iters = 10^6, ind = size(imgs)[2], indtest = size(imgstest)[2],
			n_classes = 10)

	class1 = net(["input","classifier"],
				[length(network.layers[network.nr_layers].a),n_classes],
				[0,0], [0,0])
	i2 = []
	learn_net_layerwise!(class1,i2,[iters],
	  [inputfunction for i in 1:class1.nr_layers-1],
	  [getstatichiddenrep for i in 1:class1.nr_layers-1];
	  LearningFromLayer = 2, LearningUntilLayer = 2)

	error_train = geterrors!(class1, imgs, labels; noftest = ind)
	error_test = geterrors!(class1, imgstest, labelstest; noftest = indtest)
	print(string("\n Train Accuracy: ", 100 * (1 - error_train)," % \n"))
	print(string("\n Test Accuracy: ", 100 * (1 - error_test)," % \n"))
end

##################################################################################
# Input helpers
@inline getdir() = rand([-1,1], 2)
@inline getimsize(img) = Int(sqrt(size(img)[1]))
@inline function getmovingimage(img; cut_size = 0, duration = 20, speed = 1)
	img_s = getimsize(img)
	movingimg = zeros(img_s, img_s, duration)
	dir = getdir() # select 1 of 8 possible directions
	for i in 1:duration
		movingimg[:,:,i] = circshift(reshape(img,img_s,img_s), i * speed .* dir)
	end
	return movingimg
end
@inline function getstaticimage(img; cut_size = 0)
	img_s = getimsize(img)
	return reshape(img,img_s,img_s,1)
end
@inline function getstatichiddenrep(imgs; cut_size = 0)
	reshape(imgs,length(imgs),1,1)
end

@inline function getpatchparams(img, patch_size)
	img_s = getimsize(img)
	return img_s, rand(1:img_s-patch_size, 2)
end
@inline cutindices(patchpos, patch_size) =
	CartesianIndices((patchpos[1]:patchpos[1]+patch_size-1, patchpos[2]:patchpos[2]+patch_size-1))

@inline function getmovingimagepatch(img; cut_size = 8, duration = 20, speed = 1)
	img_s, patchpos = getpatchparams(img, cut_size)
	movingpatch = zeros(cut_size, cut_size, duration)
	dir = getdir()
	for i in 1:duration
		movingpatch[:,:,i] =
			circshift(reshape(img,img_s,img_s), i * speed .* dir)[cutindices(patchpos, cut_size)]
	end
	return movingpatch
end
@inline function getstaticimagepatch(img; cut_size = 8)
	img_s, patchpos = getpatchparams(img, cut_size)
	return reshape(reshape(img, img_s, img_s)[cutindices(patchpos, cut_size)],cut_size,cut_size,1)
end

########################
# Deprecated

@inline function generatemovingpatches(patches, layer_pre, layer_post;
	nr_presentations_per_patch = 30, number_of_patches = Int(5e4), speed = 1.)
		dim = Int(ceil(sqrt(size(patches)[1])))
		movingpatches = zeros(Int(size(patches)[1]),Int(nr_presentations_per_patch*number_of_patches))
		hiddenreps = zeros(length(layer_post.a),Int(nr_presentations_per_patch*number_of_patches))
		@showprogress for i in 1:number_of_patches
			index = rand(1:Int(size(patches)[2]))
			dir = rand(-1:1,2) # select 1 of 9 possible directions
			for j in 1:nr_presentations_per_patch
				movingpatches[:,(i-1)*nr_presentations_per_patch+j] = circshift(reshape(patches[:,index],dim,dim),j * speed .* dir)[:]
				layer_pre.a = movingpatches[:,(i-1)*nr_presentations_per_patch+j]
				forwardprop_lc!(layer_pre, layer_post)
				hiddenreps[:,(i-1)*nr_presentations_per_patch+j] = deepcopy(layer_post.a)
			end
		end
		return movingpatches, hiddenreps
end

@inline function generatejitteredpatches(patches, layer_pre, layer_post; max_amplitude = 3,
	nr_presentations_per_patch = 30, number_of_patches = Int(5e4))
		dim = Int(ceil(sqrt(size(patches)[1])))
		jitteredpatches = zeros(Int(size(patches)[1]),Int(nr_presentations_per_patch*number_of_patches))
		hiddenreps = zeros(length(layer_post.a),Int(nr_presentations_per_patch*number_of_patches))
		@showprogress for i in 1:number_of_patches
			index = rand(1:Int(size(patches)[2]))
			for j in 1:nr_presentations_per_patch
				amps = rand(-max_amplitude:max_amplitude,2) #draw random translation
				jitteredpatches[:,(i-1)*nr_presentations_per_patch+j] = circshift(reshape(patches[:,index],dim,dim),amps)[:]
				layer_pre.a = jitteredpatches[:,(i-1)*nr_presentations_per_patch+j]
				forwardprop_lc!(layer_pre, layer_post)
				hiddenreps[:,(i-1)*nr_presentations_per_patch+j] = deepcopy(layer_post.a)
			end
		end
		return jitteredpatches, hiddenreps
end

#return image patches for training patchy sparse layer
@inline function cut_image_to_patches(image, layer::layer_sparse_patchy)
	full_edge_length = size(image)[1]
	patch_edge_length = layer.parameters.patch_size
	overlap = parameters.overlap
  number_of_patches_along_edge = Int(32/(patch_edge_length-overlap)-1)
	number_of_patches_along_edge^2 != layer.parameters.n_of_layer_patches ?
		error("patches/image size not compatible with layer_sparse_patches (patchsize/overlap)") : Void
  patches = zeros(patch_edge_length,patch_edge_length,number_of_patches_along_edge^2)
  for i in 1:number_of_patches_along_edge
    for j in 1:number_of_patches_along_edge
      patches[:,:,(i-1)*number_of_patches_along_edge+j] =
      image[(i-1)*(patch_edge_length - overlap)+1:i*(patch_edge_length) - (i-1)*overlap,
              (j-1)*(patch_edge_length - overlap)+1:j*(patch_edge_length) - (j-1)*overlap]
    end
  end
  return patches
end

##################################################################################
# Loss calculators


@inline function _evaluate_errors(layer_pre, layer_post, i)
	generatehiddenreps(layer_pre, layer_post)
	return [i,mean((smallimgs[:,1:Int(5e4)] - BLAS.gemm('T', 'N', layer_post.w, layer_post.hidden_reps)).^2)]
end

#up to now: only squared reconstruction error!
#for SC: full loss function:
#losses[i] = squared_errors[i] + sum(layer_post.a)-length(layer_post.a)*p + sum(layer_post.a*layer_post.a')-length(layer_post.a)*p^2
@inline function evaluate_loss(layer_pre, layer_post, i, iterations, nr_evaluations, squared_errors)
	if i == 1
		squared_errors[:,1] = _evaluate_errors(layer_pre,layer_post,i)
	elseif i % Int(iterations/nr_evaluations) == 0
		squared_errors[:,Int(i*nr_evaluations/iterations)+1] = _evaluate_errors(layer_pre,layer_post,i)
	end
end

# to evaluate difference between pure feedforward and recurrent sparse coding feed-forward
@inline function evaluate_ff_difference(layer_pre, layer_post::layer_sparse)
	layer_post.u-BLAS.gemv('N',layer_post.w,layer_pre.a)
end

@inline function gethighestvalues(array; number = 0.1*length(array))
	array1 = deepcopy(array)
  indices = []
	for i in 1:number
		index = findmax(array1)[2]
		push!(indices,index)
		array1[index] = -Inf64
	end
	return indices
end
@inline function getsomepeaks(array; factor = 1.0)
	mean_value = mean(array)
	std_value = std(array)
	indices = []
	for i in 1:length(array)
		if array[i] >= factor*std_value+mean_value push!(indices,i) end
	end
	return indices
end
@inline function getsomenegativepeaks(array; factor = 0.5)
	mean_value = mean(array)
	std_value = std(array)
	indices = []
	for i in 1:length(array)
		if array[i] <= mean_value-factor*std_value push!(indices,i) end
	end
	return indices
end

@inline function microsaccade(imagevector; max_amplitude = 3)
	dim = Int(sqrt(length(imagevector)))
	amps = rand(-max_amplitude:max_amplitude,2) #draw random translation
	circshift(reshape(imagevector,dim,dim),amps)[:]
end

#######################################################################################
# Initialization

@inline function set_init_bars!(layer::layer_sparse,hidden_size; reinit_weights = false,
		p = 1/hidden_size, one_over_tau_a = 1/1000, activationfunction = sigm!) #inspired by Földiak 1991 init
	layer.parameters.learningrate_v = 1e-1
  layer.parameters.learningrate_w = 2e-2 #1e-2
  layer.parameters.learningrate_thr = 2e-2 #5e-2 speeds up convergence
	layer.parameters.p = p
	layer.parameters.one_over_tau_a = one_over_tau_a
	layer.parameters.activationfunction = activationfunction
	reinit_weights && (layer.w = rand(size(layer.w)[1],size(layer.w)[2])/hidden_size)
end
@inline function set_init_bars!(layer::layer_sparse_patchy, hidden_size; reinit_weights = false,
		p = 1/hidden_size, one_over_tau_a = 1/1000, activationfunction = pwl!)
	for layer_patch in layer.layer_patches
		set_init_bars!(layer_patch,hidden_size_sparse; reinit_weights = reinit_weights,
			p = p, one_over_tau_a = one_over_tau_a, activationfunction = activationfunction)
	end
end

@inline function set_init_bars!(layer::layer_pool; reinit_weights = false, p = 1/2,
		one_over_tau_a = 1/8, updaterule = GH_SFA_Sanger!, activationfunction = lin!)
	layer.parameters.activationfunction = activationfunction #"relu" #pwl & relu works nice but no idea why!
	layer.parameters.updaterule = updaterule
	layer.parameters.learningrate = 1e-2 # for non lc-learning
	layer.parameters.learningrate_v = 1e-2#1e-2#5e-2#5e-2#1e-1
    layer.parameters.learningrate_w = 2e-2#2e-3#5e-3#1e-2#2e-2 WTA: 1e-2
    layer.parameters.learningrate_thr = 2e-2#2e-3#2e-4#1e-2#2e-2 stable #5e-2 speeds up convergence
	layer.parameters.one_over_tau_a = one_over_tau_a # shorter pooling time constant to not pool everything
	layer.parameters.p = p
	reinit_weights && (layer.w = rand(size(layer.w)[1],size(layer.w)[2])/size(layer.w)[1])
	#reinit_weights ? layer.v = rand(size(layer.v)[1],size(layer.v)[1]) : Void
end
@inline function set_init_bars!(layer::layer_pool_patchy;  reinit_weights = false,
		p = 1/2, one_over_tau_a = 1/8, updaterule = GH_SFA_Sanger!, activationfunction = lin!)
	for layer_patch in layer.layer_patches
		set_init_bars!(layer_patch; reinit_weights = reinit_weights, p = p,
			one_over_tau_a = one_over_tau_a, updaterule = updaterule,
			activationfunction = activationfunction)
	end
end

@inline function cutweights!(network; number = 10)
	for i in 1:size(network.layers[2].w)[1]
		#indices = gethighestvalues(abs.(network.layers[2].w[i,:]); number = number)
		indices = gethighestvalues(network.layers[2].w[i,:]; number = number)
		#indices = findall(x -> (x > 0),network.layers[2].w[i,:])
		for j in 1:size(network.layers[2].w)[2]
			network.layers[2].w[i,j] *= Int(j in indices)
		end
	end
end

#######################################################################################

function loadsharedweights!(layer::layer,filepath)
	layer = deepcopy(load(filepath,"layer"))
end
function loadsharedweights!(layer::layer_patchy,filepath)
	singlepatchlayer = load(filepath,"layer")
	for i in 1:layer.parameters.n_of_layer_patches
		layer.layer_patches[i] = deepcopy(singlepatchlayer)
	end
end
