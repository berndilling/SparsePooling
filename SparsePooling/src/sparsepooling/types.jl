
############################################################################
# Basic types and constructors
############################################################################

abstract type layer end
export layer
abstract type layer_patchy end
export layer_patchy
############################################################################
# LAYERS
mutable struct parameters_input
	one_over_tau_a::Float64
end
export parameters_input

mutable struct layer_input <: layer
	parameters::parameters_input
	a::Array{Float64, 1} #activation
	a_tr::Array{Float64, 1} #low pass filtered activity: "trace" (current time step is left out)
end
export layer_input

mutable struct layer_input_color <: layer
	parameters::parameters_input
	a::Array{Float64, 2}
	a_tr::Array{Float64, 2}
end
export layer_input_color

mutable struct parameters_sparse
	learningrate_v::Float64 # learning rate for lateral inhibition
	learningrate_w::Float64 # learning rate for feedforward weights
	learningrate_thr::Float64 # learning rate for thresholds
	dt::Float64 #time step for integration
	epsilon::Float64 #convergence criterium for recurrence
	activationfunction::Function #"relu"/"resqroot"/"heavyside"/"pwl"/"sigm"/"LIF"
	OneOverMaxFiringRate::Float64 #used for LIF nonlin. determines refractory period/saturation
	calculate_trace::Bool
	one_over_tau_a::Float64 # time constant for low pass filtered activity "trace"
	one_over_tau_a_s::Float64
	one_over_tau_a_l::Float64
	p::Float64 #average activation/"firing rate"
end
export parameters_sparse

mutable struct layer_sparse <: layer
	parameters::parameters_sparse
	a_pre::Array{Float64, 3} #activation of pre layer (3d array (ksize, ksize, nfilters))
	a_tr_pre::Array{Float64, 3} #activation-trace of pre layer
	u::Array{Float64, 1} #membrane potential
	a::Array{Float64, 1} #activation = nonlinearity(membrane potential)
	a_tr::Array{Float64, 1} #low pass filtered activity: "trace" (current time step is left out)
	a_tr_s::Array{Float64, 1} #short trace for time scale separation
	a_tr_l::Array{Float64, 1} #long trace (running average) for decorrelation learning rule
	w::Array{Float64, 2} #synaptic weight matrix in format TOxFROM
	v::Array{Float64, 2} #recurrent/lateral inhibition weight matrix
	t::Array{Float64, 1} #thresholds
	hidden_reps::Array{Float64, 2} #hidden representations of the layer (saved to accelerate further learning)
end
export layer_sparse

mutable struct parameters_sparse_patchy
	n_of_layer_patches::Int64 #number of independent sparse layer patches/populations
	patch_size::Int64 #linear dimension of patch/kernel in #pixels or #populations
	stride::Int64 #stride of patch centers (in #pixels (input) or #populations)
	in_size::Int64 #size of total input image/filter bank (1 filter): in_size*in_size
	in_fan::Int64 #total number of inputs per population
	weight_sharing::Bool
end
export parameters_sparse_patchy
mutable struct layer_sparse_patchy <: layer_patchy
	parameters::parameters_sparse_patchy
	layer_patches::Array{layer_sparse, 1}
	a::Array{Float64, 1} #combined sparse activity of all sparse layer patches
	a_tr::Array{Float64, 1} #same for activity trace
	a_max::Float64 #max activation of whole layer (for normalization)
end
export layer_sparse_patchy

mutable struct parameters_pool
	learningrate::Float64
	learningrate_v::Float64 # learning rate for lateral inhibition
	learningrate_w::Float64 # learning rate for feedforward weights
	learningrate_thr::Float64 # learning rate for thresholds
	dt::Float64 #time step for integration
	epsilon::Float64 #convergence criterium for recurrence
	updaterule::Function #"Sanger"/"Oja"
	activationfunction::Function
	calculate_trace::Bool
	one_over_tau_a ::Float64
	one_over_tau_a_l ::Float64
	p::Float64
end
export parameters_pool

mutable struct layer_pool <: layer
	parameters::parameters_pool
	a_pre::Array{Float64, 3} #activation of pre layer (3d array (ksize, ksize, nfilters))
	a_tr_pre::Array{Float64, 3} #activation-trace of pre layer
	a_tr_s_pre::Array{Float64, 3}
	u::Array{Float64, 1} #membrane potential
	a::Array{Float64, 1} #activation = nonlinearity(membrane potential)
	a_tr::Array{Float64, 1} #low pass filtered activity: "trace" (current time step is left out)
	a_tr_l::Array{Float64, 1} #long trace (running average) for decorrelation learning rule
	w::Array{Float64, 2} #synaptic weight matrix in format TOxFROM
	v::Array{Float64, 2} #recurrent/lateral inhibition weight matrix
	t::Array{Float64, 1} #thresholds
	#b::Array{Float64, 1} #biases
	hidden_reps::Array{Float64, 2} #hidden representations of the layer
end
export layer_pool

mutable struct layer_max_pool <: layer
	a_pre::Array{Float64, 3} #activation of pre layer (3d array (ksize, ksize, nfilters))
	a_tr_pre::Array{Float64, 3} #activation-trace of pre layer
	a::Array{Float64, 1} # pooled activity
	a_tr::Array{Float64, 1} # pooled ...
end
export layer_max_pool

mutable struct parameters_pool_patchy
	n_of_layer_patches::Int64 #number of independent pool layer patches/populations
	patch_size::Int64 #linear dimension of patch/kernel in #pixels or #populations
	stride::Int64 #stride of patch centers (in #pixels (input) or #populations)
	in_size::Int64 #size of total image/filter bank (1 filter): in_size*in_size
	in_fan::Int64 #total number of inputs per population
	weight_sharing::Bool
end
export parameters_pool_patchy
mutable struct layer_pool_patchy{Tparams} <: layer_patchy
	parameters::Tparams
	layer_patches::Array{<:layer, 1}
	a::Array{Float64, 1} #combined activity of all sparse layer patches
	a_tr::Array{Float64, 1} #same for activity trace
end
export layer_pool_patchy

# Supervised classifier on the activations of a "net" (could access multiple levels of hierarchy!)
mutable struct classifier <: layer
	nl::Int64 #number of layers in classifier (without input layer which is part of the net)
	a_pre::Array{Float64, 1} #activation of pre layer (needed for parallel/patchy)
	a_tr_pre::Array{Float64, 1} #activation-trace of pre layer
	u::Array{Array{Float64, 1}, 1} #membrane potential
	a::Array{Array{Float64, 1}, 1} #activation = nonlinearity(membrane potential)
	e::Array{Array{Float64, 1}, 1} #error with respect to target
	w::Array{Array{Float64, 2}, 1} #synaptic weight matrix
	b::Array{Array{Float64, 1}, 1} #biases
	activationfunctions::Array{Function, 1} #nonlinearities for every layer
end
export classifier

############################################################################
## NETWORKS

# Network containing multiple layers.
# Tlayers has to be a tuple of layers. For constructing, e.g. net(... , (layer_input(...), layer_sparse(...)))
# If only one layer is presented: pass (layer,) as argument
mutable struct net{Tlayers}
	nr_layers::Int64 #number of layers
	layer_types::Array{String, 1} #type of layers: sparse, sparse_patchy or pool (...maybe others later)
	layer_sizes::Array{Int64, 1} #sizes of layers (number of neurons)
	layer_kernelsizes::Array{Int64, 1} #size (linear dim) of kernel (0 for fully connected)
	layer_strides::Array{Int64, 1} #strides (0 for fully connected)
	layers::Tlayers #layers of the network
end
export net

############################################################################
## DATA

mutable struct labelleddata{Tdata, Tclasses, Tlabels}
	data::Tdata
	classes::Tclasses
	labels::Tlabels
	nsamples::Int64
	currentsample::Int64
	margin::Int64
	color::Bool
end
labelleddata(data, labels; classes = 0:9, margin = 0, color = false) = labelleddata(data, classes, labels, size(data)[end], 0, margin, color)
export labelleddata

mutable struct NORBdata{Tdata, Tclasses, Tlabels, Tinstlist, Televlist, Tazlist, Tlightlist}
	data::Tdata
	classes::Tclasses
    labels::Tlabels
	instance_list::Tinstlist
	elevation_list::Televlist
    azimuth_list::Tazlist
	lighting_list::Tlightlist
	nsamples::Int64
	currentsample::Int64
end
NORBdata(data, category_list, instance_list, elevation_list, azimuth_list, lighting_list; classes = 0:4) =
	NORBdata(data, classes, category_list, instance_list, elevation_list, azimuth_list, lighting_list, size(data)[end], 0)
export NORBdata
getindim(data::labelleddata) = size(data.data)[1]
getindim(data::NORBdata) = size(data.data)[1]^2
export getindim

############################################################################
## CONSTRUCTORS
############################################################################

function parameters_input(; one_over_tau_a = 1e-3)
	parameters_input(one_over_tau_a)
end
function layer_input(ns::Int64) #ns: number of neurons in input layer
	layer_input(parameters_input(), # default parameter init
	zeros(ns),
	zeros(ns))
end
function layer_input_color(ns::Int64)
	layer_input_color(parameters_input(),
	zeros(ns, 3),
	zeros(ns, 3))
end

function parameters_sparse(ns; learningrate_v = 1e-1, learningrate_w = 5e-3, learningrate_thr = 5e-2, # 1e-1, 5e-3, 5e-2
		dt = 1e-2, epsilon = 1e-2, activationfunction = SparsePooling.relu!, OneOverMaxFiringRate = 1/50, # sigm_m!
		calculate_trace = true, one_over_tau_a = 1e-2, one_over_tau_a_s = 1., one_over_tau_a_l = 1e-2,
		p = 1. / ns[2]) #p = 1/12 average activation set to 5% (as in Zylberberg)
	parameters_sparse(learningrate_v, learningrate_w, learningrate_thr,
			dt, epsilon, activationfunction, OneOverMaxFiringRate,
			calculate_trace, one_over_tau_a, one_over_tau_a_s, one_over_tau_a_l, p)
end
function layer_sparse(ns::Array{Int64, 1}; ksize = 1, n_in_channel = ns[1], one_over_tau_a = 1e-2, p = 1. / ns[2])
	#ns: number of neurons/filters in previous and present layer
	in_fan = ksize ^ 2 * n_in_channel
	layer_sparse(parameters_sparse(ns; one_over_tau_a = one_over_tau_a, p = p), # default parameter init
			zeros(ksize, ksize, n_in_channel), #pre activation initialized with zeros
			zeros(ksize, ksize, n_in_channel), #pre low-pass filtered activity initialized with zeros
			zeros(ns[2]), #membrane potential initialized with zeros
			zeros(ns[2]), #activation initialized with zeros
			zeros(ns[2]), #low-pass filtered activity initialized with zeros
			zeros(ns[2]), #a_tr_s
			zeros(ns[2]), #a_tr_l
			randn(ns[2], in_fan)/(10*sqrt(in_fan)), #feed-forward weights initialized gaussian distr.
			zeros(ns[2], ns[2]), #lateral inhibition initialized with zeros
			5*ones(ns[2]), #thresholds initialized with 5's (as in Zylberberg) (zero maybe not so smart...)
			zeros(ns[2],1)) #reps initialized with zeros (only 1 reps here, but can be changed later)
end
function get_n_of_layer_patches(in_size::Int64, patch_size::Int64, stride::Int64)
	# in_size: linear number of dimension (pixel/populations) in previous layer
	# patch_size: linear dimension of patch/kernel in pixels/populations
	# stride: stride of patches/kernel in #pixels or #populations
	Int(floor((in_size - patch_size) / stride) + 1) ^ 2
end
function init_weight_sharing!(layer::layer_patchy)
	for layer_patch in layer.layer_patches
		layer_patch.w = layer.layer_patches[1].w
		layer_patch.v = layer.layer_patches[1].v
		layer_patch.t = layer.layer_patches[1].t
		layer_patch.parameters.learningrate_v /= layer.parameters.n_of_layer_patches
		layer_patch.parameters.learningrate_w /= layer.parameters.n_of_layer_patches
		layer_patch.parameters.learningrate_thr /= layer.parameters.n_of_layer_patches
	end
end
function layer_sparse_patchy(ns::Array{Int64, 1};
		patch_size = 10, n_in_channel = ns[1], stride = 1, in_size = 32,
		n_of_layer_patches = get_n_of_layer_patches(in_size, patch_size, stride),
		one_over_tau_a = 1e-2, p = 1. / ns[2],
		weight_sharing = false,
		maxpool = false)
	in_fan = patch_size ^ 2 * n_in_channel
	layer = layer_sparse_patchy(parameters_sparse_patchy(n_of_layer_patches, patch_size, stride, in_size, in_fan, weight_sharing),
		[layer_sparse(ns; ksize = patch_size, n_in_channel = n_in_channel, one_over_tau_a = one_over_tau_a, p = p) for i in 1:n_of_layer_patches],
		zeros(ns[2]*n_of_layer_patches),
		zeros(ns[2]*n_of_layer_patches),
		1.)
	weight_sharing && init_weight_sharing!(layer)
	return layer
end

function parameters_pool(ns; learningrate = 1e-2, learningrate_v = 1e-1, learningrate_w = 5e-3, learningrate_thr = 5e-2,
		dt = 1e-2, epsilon = 1e-2, updaterule = SparsePooling.GH_SFA_subtractrace_Sanger!,
	activationfunction = SparsePooling.relu!, calculate_trace = true, one_over_tau_a = 2e-1, one_over_tau_a_l = 1e-2,  p = 1. / ns[2]) # sigm_m!, p = 1/2 # one_over_tau_a = 1e-2
	parameters_pool(learningrate, learningrate_v, learningrate_w, learningrate_thr,
			dt, epsilon, updaterule, activationfunction, calculate_trace, one_over_tau_a, one_over_tau_a_l, p)
end
function layer_pool(ns::Array{Int64, 1}; ksize = 1, n_in_channel = ns[1], one_over_tau_a = 2e-1, p = 1. / ns[2])
	in_fan = ksize ^ 2 * n_in_channel
	layer_pool(parameters_pool(ns; one_over_tau_a = one_over_tau_a, p = p), # default parameter init
			zeros(ksize, ksize, n_in_channel), #pre activation initialized with zeros
			zeros(ksize, ksize, n_in_channel), #pre low-pass filtered activity initialized with zeros
			zeros(ksize, ksize, n_in_channel), #pre a_tr_s
			zeros(ns[2]), #membrane potential initialized with zeros
			zeros(ns[2]), #activation initialized with zeros
			zeros(ns[2]), #low-pass filtered activity initialized with zeros
			zeros(ns[2]), #a_tr_l
			randn(ns[2], in_fan)/(10*sqrt(in_fan)), #feed-forward weights initialized gaussian distr. # rand(ns[2], ns[1])/(10*sqrt(ns[1])),#
			zeros(ns[2], ns[2]), #lateral inhibition initialized with zeros
			5*ones(ns[2]), #thresholds initialized with zeros
			#zeros(ns[2]), # biases equal zero for linear computation such as PCA! OR rand(ns[2])/10) #biases initialized equally distr.
			zeros(ns[2],1)) #reps initialized with zeros (only 1 reps here, but can be changed later)
end
function layer_max_pool(ns::Array{Int64, 1}; ksize = 1, n_in_channel = ns[1])
	layer_max_pool(zeros(ksize, ksize, n_in_channel), # pre activations
	zeros(ksize, ksize, n_in_channel), # pre low pass filtered activations
	zeros(ns[2]), # pooled activations
	zeros(ns[2])) # pooled ...
end
function layer_pool_patchy(ns::Array{Int64, 1};
		patch_size = 10, n_in_channel = ns[1], stride = 1, in_size = 32,
		n_of_layer_patches = get_n_of_layer_patches(in_size, patch_size, stride),
		one_over_tau_a = 2e-1, p = 1. / ns[2],
		weight_sharing = false,
		maxpool = false)
	in_fan = patch_size ^ 2 * n_in_channel
	if maxpool
		layer_patches = [layer_max_pool(ns; ksize = patch_size, n_in_channel = n_in_channel) for i in 1:n_of_layer_patches]
	else
		layer_patches = [layer_pool(ns; ksize = patch_size, n_in_channel = n_in_channel, one_over_tau_a = one_over_tau_a, p = p) for i in 1:n_of_layer_patches]
	end
	layer = layer_pool_patchy(parameters_pool_patchy(n_of_layer_patches, patch_size, stride, in_size, in_fan, weight_sharing),
		layer_patches,
		zeros(ns[2]*n_of_layer_patches),
		zeros(ns[2]*n_of_layer_patches))
	weight_sharing && init_weight_sharing!(layer)
	return layer
end


function classifier(ns::Array{Int64, 1}) #ns: array of layer sizes in classifier
	nl = length(ns)
	classifier(nl - 1,
			zeros(ns[1]),
			zeros(ns[1]),
		    [zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[rand(ns[i])/10 for i in 2:nl],
			[relu! for i in 2:nl])
end

function addfullyconnectedlayer!(layers, i, layertype, tl, sl, taus, ps)
	if tl[i-1] == "sparse_patchy" || tl[i-1] == "pool_patchy" || tl[i-1] == "max_pool_patchy"
		if tl[i] == "classifier"
			layers = (layers... , layertype(vcat(layers[i-1].parameters.n_of_layer_patches * sl[i-1], sl[i:end])))
		else
			layers = (layers... , layertype(sl[i-1:i]; ksize = Int(sqrt(layers[i-1].parameters.n_of_layer_patches)),
											one_over_tau_a = 1. / taus[i], p = ps[i]))
		end
	elseif tl[i-1] == "input" || tl[i-1] == "input_color"
		if tl[i] == "classifier"
			layers = (layers... , layertype(sl[i-1:end]))
		else
			(tl[i-1] == "input") ? (n_in_channel = 1) : (n_in_channel = 3)
			layers = (layers... , layertype(sl[i-1:i]; ksize = Int(sqrt(layers[i-1].parameters.n_of_layer_patches)),
							n_in_channel = n_in_channel, one_over_tau_a = 1. / taus[i], p = ps[i]))
		end
	else
		if tl[i] == "classifier"
			layers = (layers... , layertype(sl[i-1:end]))
		else
			layers = (layers... , layertype(sl[i-1:i]; ksize = 1,
											one_over_tau_a = 1. / taus[i], p = ps[i]))
		end
	end
	return layers
end
function addpatchylayer!(layers, i, layertype, tl, sl, ks, str, taus, ps;
						weight_sharing = false, maxpool = false)
	if tl[i-1] == "sparse_patchy" || tl[i-1] == "pool_patchy" || tl[i-1] == "max_pool_patchy"
		in_size = Int(sqrt(layers[i-1].parameters.n_of_layer_patches))
		n_in_channel = sl[i-1]
	elseif tl[i-1] == "input" || tl[i-1] == "input_color"
		(tl[i-1] == "input") ? (n_in_channel = 1) : (n_in_channel = 3) # 3 channles for color images
		in_size = Int(sqrt(length(layers[i-1].a) / n_in_channel))
	end
	layers = (layers... , layertype(sl[i-1:i];
				patch_size = ks[i], n_in_channel = n_in_channel, stride = str[i],
				in_size = in_size, one_over_tau_a = 1. / taus[i], p = ps[i],
				weight_sharing = weight_sharing, maxpool = maxpool))
	return layers
end
function net(tl::Array{String, 1}, # tl: types of layers
			 sl::Array{Int64, 1}, # sl: number of neurons (patchy layers: per popul.)
			 ks::Array{Int64, 1}, # ks: kernel/patch sizes
			 str::Array{Int64, 1}, # str: strides
			 taus::Array{Float64, 1}, # time scales tau for SFA (ignored for non SFA layers)
			 ps::Array{Float64, 1}; # ps: sparsity parameters p
			 weight_sharing = false)
	nl = length(tl)
	layers = ()
	for i in 1:nl
		if tl[i] == "input"
			layers = (layers... , layer_input(sl[i]))
		elseif tl[i] == "input_color"
			layers = (layers... , layer_input_color(sl[i]))
		elseif tl[i] == "sparse"
			layers = addfullyconnectedlayer!(layers, i, layer_sparse, tl, sl, taus, ps)
		elseif tl[i] == "pool"
			layers = addfullyconnectedlayer!(layers, i, layer_pool, tl, sl, taus, ps)
		elseif tl[i] == "sparse_patchy"
			layers = addpatchylayer!(layers, i, layer_sparse_patchy, tl, sl, ks, str, taus, ps; weight_sharing = weight_sharing)
		elseif tl[i] == "pool_patchy"
			layers = addpatchylayer!(layers, i, layer_pool_patchy, tl, sl, ks, str, taus, ps; weight_sharing = weight_sharing)
		elseif tl[i] == "max_pool_patchy"
			layers = addpatchylayer!(layers, i, layer_pool_patchy, tl, sl, ks, str, taus, ps; maxpool = true)
		elseif tl[i] == "classifier"
			layers = addfullyconnectedlayer!(layers, i, classifier, tl, sl, taus, ps)
			return net(nl, tl, sl, ks, str, layers)
		end
	end
	net(nl, tl, sl, ks, str, layers)
end
export net
