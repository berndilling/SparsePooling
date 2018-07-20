
using Distributions, ProgressMeter, JLD, HDF5

#####################################################
#Helpers

function getsparsity(input::Array{Float64, 1})
	length(find(x -> (x == 0),input))/length(input)
end

function getsmallimg()
    patternindex = rand(1:size(smallimgs)[2])
    smallimgs[:, patternindex]
end

function getsmallimg(iteration) #select images in fixed order
		if iteration > size(smallimgs)[2] iteration = (iteration % size(smallimgs)[2])+1 end
    smallimgs[:, iteration]
end

function getsample()
    patternindex = rand(1:n_samples)
    smallimgs[:, patternindex]
end

function getlabel(x)
    [labels[patternindex] == i for i in 0:9]
end


function getsavepath()
	if is_apple()
		path = "/Users/Bernd/Documents/PhD/Projects/"
	elseif is_linux()
		path = "/home/illing/"
	end
end

function assigninput!(layer,images,i)
	input = images[:,i]
	if norm(input) == 0
		assigninput!(layer,images,i+1)
	else
		layer.a = input
	end
end
function generatehiddenreps(layer_pre, layer_post, images; number_of_reps = Int(5e4), mode = "no_lc")
	print("\n")
	print(string("Generate ",number_of_reps," hidden representations for layer type: ",typeof(layer_post)))
	print("\n")
	layer_post.hidden_reps = zeros(length(layer_post.a),number_of_reps)
	@showprogress for i in 1:number_of_reps
		assigninput!(layer_pre,images,i)
		if mode == "lc"
			forwardprop_lc!(layer_pre, layer_post)
		else
			#print(i)
			forwardprop!(layer_pre, layer_post)
		end
		layer_post.hidden_reps[:,i] = deepcopy(layer_post.a)
	end
end

function generateratetriggeredrecfields(layer_pre, layer_post; number_of_reps = Int(5e4), mode = "no_lc")
	#smallimgs in main have to be (sparse coding) hidden reps of jitteredpatches
	print("\n")
	print(string("Generate rate triggered receptive fields for layer type: ",typeof(layer_post)))
	print("\n")
	ratetriggeredrecfields = zeros(length(layer_post.a),length(layer_pre.a))
	@showprogress for i in 1:number_of_reps
		layer_pre.a = smallimgs[:,i]
		if mode == "lc"
			forwardprop_lc!(layer_pre, layer_post)
		else
			forwardprop!(layer_pre, layer_post)
		end
		ratetriggeredrecfields += deepcopy(layer_post.a*layer_pre.a')
	end
	ratetriggeredrecfields./number_of_reps
end

function generatecomplexrecfields(layer_pre,layer_post,jitteredpatches; mode = "no_lc")
	#smallimgs in main have to be (sparse coding) hidden reps of jitteredpatches
	print("\n")
	print(string("Generate complex rec. fields for layer type: ",typeof(layer_post)))
	print("\n")
	print("CAUTION: smallimgs in main have to be sparse coding hidden reps!")
	print("\n")
	complexrecfields = zeros(length(layer_post.a),size(jitteredpatches)[1])
	@showprogress for i in 1:size(jitteredpatches)[2]
		layer_pre.a = smallimgs[:,i]
		if mode == "lc"
			forwardprop_lc!(layer_pre, layer_post)
		else
			forwardprop!(layer_pre, layer_post)
		end
		complexrecfields += deepcopy(layer_post.a*jitteredpatches[:,i]')
	end
	complexrecfields./size(jitteredpatches)[2]
end

##################################################################################
# Input helpers

function generatemovingpatches(patches, layer_pre, layer_post;
	nr_presentations_per_patch = 30, number_of_patches = Int(5e4))
		dim = Int(ceil(sqrt(size(patches)[1])))
		movingpatches = zeros(Int(size(patches)[1]),Int(nr_presentations_per_patch*number_of_patches))
		hiddenreps = zeros(length(layer_post.a),Int(nr_presentations_per_patch*number_of_patches))
		@showprogress for i in 1:number_of_patches
			index = rand(1:Int(size(patches)[2]))
			dir = rand(-1:1,2) # select 1 of 9 possible directions
			for j in 1:nr_presentations_per_patch
				movingpatches[:,(i-1)*nr_presentations_per_patch+j] = circshift(reshape(patches[:,index],dim,dim),j.*dir)[:]
				layer_pre.a = movingpatches[:,(i-1)*nr_presentations_per_patch+j]
				forwardprop_lc!(layer_pre, layer_post)
				hiddenreps[:,(i-1)*nr_presentations_per_patch+j] = deepcopy(layer_post.a)
			end
		end
		return movingpatches, hiddenreps
end

function generatejitteredpatches(patches, layer_pre, layer_post; max_amplitude = 3,
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
function cut_image_to_patches(image, layer::layer_sparse_patchy)
	full_edge_length = size(image)[1]
	patch_edge_length = layer.parameters.patch_size
	overlap = parameters.overlap
  number_of_patches_along_edge = Int(32/(patch_edge_length-overlap)-1)
	number_of_patches_along_edge^2 != layer.parameters.n_of_sparse_layer_patches ?
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


function _evaluate_errors(layer_pre, layer_post, i)
	generatehiddenreps(layer_pre, layer_post)
	return [i,mean((smallimgs[:,1:Int(5e4)] - BLAS.gemm('T', 'N', layer_post.w, layer_post.hidden_reps)).^2)]
end

#up to now: only squared reconstruction error!
#for SC: full loss function:
#losses[i] = squared_errors[i] + sum(layer_post.a)-length(layer_post.a)*p + sum(layer_post.a*layer_post.a')-length(layer_post.a)*p^2
function evaluate_loss(layer_pre, layer_post, i, iterations, nr_evaluations, squared_errors)
	if i == 1
		squared_errors[:,1] = _evaluate_errors(layer_pre,layer_post,i)
	elseif i % Int(iterations/nr_evaluations) == 0
		squared_errors[:,Int(i*nr_evaluations/iterations)+1] = _evaluate_errors(layer_pre,layer_post,i)
	end
end

# to evaluate difference between pure feedforward and recurrent sparse coding feed-forward
function evaluate_ff_difference(layer_pre, layer_post::layer_sparse)
	layer_post.u-BLAS.gemv('N',layer_post.w,layer_pre.a)
end

function gethighestvalues(array; number = 0.1*length(array))
	array1 = deepcopy(array)
  indices = []
	for i in 1:number
		index = findmax(array1)[2]
		push!(indices,index)
		array1[index] = -Inf64
	end
	return indices
end
function getsomepeaks(array; factor = 1.0)
	mean_value = mean(array)
	std_value = std(array)
	indices = []
	for i in 1:length(array)
		if array[i] >= factor*std_value+mean_value push!(indices,i) end
	end
	return indices
end
function getsomenegativepeaks(array; factor = 0.5)
	mean_value = mean(array)
	std_value = std(array)
	indices = []
	for i in 1:length(array)
		if array[i] <= mean_value-factor*std_value push!(indices,i) end
	end
	return indices
end

function microsaccade(imagevector; max_amplitude = 3)
	dim = Int(sqrt(length(imagevector)))
	amps = rand(-max_amplitude:max_amplitude,2) #draw random translation
	circshift(reshape(imagevector,dim,dim),amps)[:]
end

function set_init_bars!(layer::layer_sparse,hidden_size)
	layer.w = rand(size(layer.w)[1],size(layer.w)[2])/hidden_size
	layer.parameters.learningrate_v = 1e-1
  layer.parameters.learningrate_w = 2e-2
  layer.parameters.learningrate_thr = 2e-2
end

function set_init_bars!(layer::layer_pool)
	layer.parameters.activationfunction = lin!#"relu" #pwl & relu works nice but no idea why!
	layer.parameters.updaterule = GH_SFA_Sanger!
	layer.parameters.learningrate = 1e-2
	layer.parameters.one_over_tau_a = 1e-1
end

function cutweights!(network; number = 10)
	for i in 1:size(network.layers[2].w)[1]
		#indices = gethighestvalues(abs.(network.layers[2].w[i,:]); number = number)
		indices = gethighestvalues(network.layers[2].w[i,:]; number = number)
		#indices = find(x -> (x > 0),network.layers[2].w[i,:])
		for j in 1:size(network.layers[2].w)[2]
			network.layers[2].w[i,j] *= Int(j in indices)
		end
	end
end

# to save layer, take care that parameters are the same in the new net and the saved one!
function savelayer(path,layer::layer_sparse_patchy)
  layerfields = []
  for sparse_layer_patch in layer.sparse_layer_patches
    push!(layerfields, [sparse_layer_patch.u,sparse_layer_patch.a,
      sparse_layer_patch.a_tr,sparse_layer_patch.w,sparse_layer_patch.v,
      sparse_layer_patch.t,sparse_layer_patch.hidden_reps])
  end
  save(path,"sparse_layer_patches_fields",layerfields,
		"parameters_sparse_patchy",layer.parameters,
		"parameters_sparse_layer_patch",string(layer.sparse_layer_patches[1].parameters))
end
function savelayer(path,layer::layer_pool)
  layerfields = [layer.u,layer.a,
      layer.a_tr,layer.w,layer.v,
      layer.t,layer.b,layer.hidden_reps]
  save(path,"layer_fields",layerfields,
		"parameters_pool_layer",string(layer.parameters))
end
function loadlayer!(path,layer::layer_sparse_patchy)
	layerfields = load(path,"sparse_layer_patches_fields")
	n_of_sparse_patches = load(path,"parameters_sparse_patchy").n_of_sparse_layer_patches
	for i in 1:n_of_sparse_patches
		layer.sparse_layer_patches[i].u = layerfields[i][1]
		layer.sparse_layer_patches[i].a = layerfields[i][2]
		layer.sparse_layer_patches[i].a_tr = layerfields[i][3]
		layer.sparse_layer_patches[i].w = layerfields[i][4]
		layer.sparse_layer_patches[i].v = layerfields[i][5]
		layer.sparse_layer_patches[i].t = layerfields[i][6]
		layer.sparse_layer_patches[i].hidden_reps = layerfields[i][7]
	end
end
function loadlayer!(path,layer::layer_pool)
	layerfields = load(path,"layer_fields")
	layer.u = layerfields[1]
	layer.a = layerfields[2]
	layer.a_tr = layerfields[3]
	layer.w = layerfields[4]
	layer.v = layerfields[5]
	layer.t = layerfields[6]
	layer.b = layerfields[7]
	layer.hidden_reps = layerfields[8]
end
