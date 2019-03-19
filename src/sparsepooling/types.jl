
############################################################################
# Basic types and constructors
############################################################################

############################################################################
# LAYERS
mutable struct parameters_input
	one_over_tau_a::Float64
end

mutable struct layer_input
	parameters::parameters_input
	a::Array{Float64, 1} #activation
	a_tr::Array{Float64, 1} #low pass filtered activity: "trace" (current time step is left out)
end

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
	p::Float64 #average activation/"firing rate"
end

mutable struct layer_sparse
	parameters::parameters_sparse
	a_pre::Array{Float64, 1} #activation of pre layer (needed for parallel/patchy)
	a_tr_pre::Array{Float64, 1} #activation-trace of pre layer
	u::Array{Float64, 1} #membrane potential
	a::Array{Float64, 1} #activation = nonlinearity(membrane potential)
	a_tr::Array{Float64, 1} #low pass filtered activity: "trace" (current time step is left out)
	a_tr_s::Array{Float64, 1}
	w::Array{Float64, 2} #synaptic weight matrix in format TOxFROM
	v::Array{Float64, 2} #recurrent/lateral inhibition weight matrix
	t::Array{Float64, 1} #thresholds
	hidden_reps::Array{Float64, 2} #hidden representations of the layer (saved to accelerate further learning)
end

mutable struct parameters_sparse_patchy
	n_of_sparse_layer_patches::Int64 #number of independent sparse layer patches
	patch_size::Int64 #number of each in-fan: patch_size*patch_size
	overlap::Int64 #overlap of patches in pixels
	image_size::Int64 #size of total image: image_size*image_size
end
# function parameters_sparse_patchy(; n_of_sparse_layer_patches = 49,
# 	patch_size = 8, overlap = 4)
# 	parameters_sparse_patchy(n_of_sparse_layer_patches, patch_size, overlap)
# end
mutable struct layer_sparse_patchy
	parameters::parameters_sparse_patchy
	sparse_layer_patches::Array{layer_sparse, 1}
	a::Array{Float64, 1} #combined sparse activity of all sparse layer patches
	a_tr::Array{Float64, 1} #same for activity trace
end

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
	p::Float64
end

mutable struct layer_pool
	parameters::parameters_pool
	a_pre::Array{Float64, 1} #activation of pre layer (needed for parallel/patchy)
	a_tr_pre::Array{Float64, 1} #activation-trace of pre layer
	a_tr_s_pre::Array{Float64, 1}
	u::Array{Float64, 1} #membrane potential
	a::Array{Float64, 1} #activation = nonlinearity(membrane potential)
	a_tr::Array{Float64, 1} #low pass filtered activity: "trace" (current time step is left out)
	w::Array{Float64, 2} #synaptic weight matrix in format TOxFROM
	v::Array{Float64, 2} #recurrent/lateral inhibition weight matrix
	t::Array{Float64, 1} #thresholds
	b::Array{Float64, 1} #biases
	hidden_reps::Array{Float64, 2} #hidden representations of the layer
end

mutable struct parameters_pool_patchy
	n_of_pool_layer_patches::Int64 #number of independent pool layer patches
	in_fan::Int64 # number of pre-synaptic input units per patch
end
mutable struct layer_pool_patchy
	parameters::parameters_pool_patchy
	pool_layer_patches::Array{layer_pool, 1}
	a::Array{Float64, 1} #combined activity of all sparse layer patches
	a_tr::Array{Float64, 1} #same for activity trace
end

# Supervised classifier on the activations of a "net" (could access multiple levels of hierarchy!)
mutable struct classifier
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

############################################################################
## NETWORKS

#Network containing sparse and pooling layers
mutable struct net
	nr_layers::Int64 #number of layers
	layer_sizes::Array{Int64, 1} #sizes of layers (number of neurons)
	layer_types::Array{String, 1} #type of layers: sparse, sparse_patchy or pool (...maybe others later)
	layers::Array{Any, 1} #layers of the network
end

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

function parameters_sparse(; learningrate_v = 5e-2, learningrate_w = 5e-3, learningrate_thr = 5e-2,
		dt = 5e-2, epsilon = 1e-2, activationfunction = relu!, OneOverMaxFiringRate = 1/50,
		calculate_trace = true, one_over_tau_a = 1e-2,
		one_over_tau_a_s = 1.,
		p = 1/10) #p = 1/12 average activation set to 5% (as in Zylberberg)
	parameters_sparse(learningrate_v, learningrate_w, learningrate_thr,
			dt, epsilon, activationfunction, OneOverMaxFiringRate,
			calculate_trace, one_over_tau_a,
			one_over_tau_a_s,
			p)
end
function layer_sparse(ns::Array{Int64, 1}; in_fan = ns[1]) #ns: number of neurons in previous and present layer
	layer_sparse(parameters_sparse(), # default parameter init
			zeros(in_fan), #pre activation initialized with zeros
			zeros(in_fan), #pre low-pass filtered activity initialized with zeros
			zeros(ns[2]), #membrane potential initialized with zeros
			zeros(ns[2]), #activation initialized with zeros
			zeros(ns[2]), #low-pass filtered activity initialized with zeros
			zeros(ns[2]),
			randn(ns[2], in_fan)/(10*sqrt(in_fan)), #feed-forward weights initialized gaussian distr.
			zeros(ns[2], ns[2]), #lateral inhibition initialized with zeros
			5*ones(ns[2]), #thresholds initialized with 5's (as in Zylberberg) (zero maybe not so smart...)
			zeros(ns[2],1)) #reps initialized with zeros (only 1 reps here, but can be changed later)
end
function get_n_of_layer_patches(image_size, patch_size, overlap)
	(overlap == 0) ? Int(image_size/patch_size)^2 : (Int(image_size/(patch_size - overlap))-1)^2
end
function layer_sparse_patchy(ns::Array{Int64, 1};
		patch_size = 8, in_fan = patch_size^2, overlap = 4, image_size = 32, #ns: size of in-fan and hidden layer per sparse layer patch
		n_of_sparse_layer_patches = get_n_of_layer_patches(image_size, patch_size, overlap))
	layer_sparse_patchy(parameters_sparse_patchy(n_of_sparse_layer_patches, patch_size, overlap, image_size),
	[layer_sparse(ns; in_fan = in_fan) for i in 1:n_of_sparse_layer_patches],
	zeros(ns[2]*n_of_sparse_layer_patches),
	zeros(ns[2]*n_of_sparse_layer_patches)
	)
end

function parameters_pool(; learningrate = 1e-2, learningrate_v = 5e-2, learningrate_w = 5e-3, learningrate_thr = 1e-1,
		dt = 1e-2, epsilon = 1e-2, updaterule = GH_SFA_Sanger!,
	activationfunction = pwl!, calculate_trace = true, one_over_tau_a = 1e-2, p = 1/20) # p = 1/2
	parameters_pool(learningrate, learningrate_v, learningrate_w, learningrate_thr,
			dt, epsilon, updaterule, activationfunction, calculate_trace, one_over_tau_a, p)
end
function layer_pool(ns::Array{Int64, 1})
	layer_pool(parameters_pool(), # default parameter init
			zeros(ns[1]), #pre activation initialized with zeros
			zeros(ns[1]), #pre low-pass filtered activity initialized with zeros
			zeros(ns[1]),
			zeros(ns[2]), #membrane potential initialized with zeros
			zeros(ns[2]), #low-pass filtered activity initialized with zeros
			zeros(ns[2]), #activation initialized with zeros
			randn(ns[2], ns[1])/(10*sqrt(ns[1])), #feed-forward weights initialized gaussian distr. # rand(ns[2], ns[1])/(10*sqrt(ns[1])),#
			zeros(ns[2], ns[2]), #lateral inhibition initialized with zeros
			zeros(ns[2]), #thresholds initialized with zeros
			zeros(ns[2]), # biases equal zero for linear computation such as PCA! OR rand(ns[2])/10) #biases initialized equally distr.
			zeros(ns[2],1)) #reps initialized with zeros (only 1 reps here, but can be changed later)
end
function layer_pool_patchy(ns::Array{Int64, 1}; in_fan = ns[1], patch_size = 8, overlap = 4, image_size = 32, # pooling occurs within each sparse layer patch
		n_of_pool_layer_patches = get_n_of_layer_patches(image_size, patch_size, overlap))
	layer_pool_patchy(parameters_pool_patchy(n_of_pool_layer_patches, in_fan),
	[layer_pool(ns) for i in 1:n_of_pool_layer_patches],
	zeros(ns[2]*n_of_pool_layer_patches),
	zeros(ns[2]*n_of_pool_layer_patches)
	)
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

# This constructor assumes a factor-2-downsampling at each pool/sparse transition
# -> doubling of receptive field size
function net(sl::Array{Int64, 1}, tl::Array{String, 1}; overlap = false) #sl: Sizes of layers, tl: types of layers
	nl = length(sl)
	network = net(nl,sl,tl,[layer_input(sl[1])])
	for i in 2:nl
		if tl[i] == "sparse"
			if tl[i-1] == "sparse_patchy"
				push!(network.layers,layer_sparse([network.layers[i-1].parameters.n_of_sparse_layer_patches * sl[i-1],sl[i]]))
			elseif tl[i-1] == "pool_patchy"
				push!(network.layers,layer_sparse([network.layers[i-1].parameters.n_of_pool_layer_patches * sl[i-1],sl[i]]))
			else
				push!(network.layers,layer_sparse(sl[i-1:i]))
			end
		elseif tl[i] == "sparse_patchy"
			if tl[i-1] == "pool_patchy"
				push!(network.layers,layer_sparse_patchy(sl[i-1:i];
					patch_size = 2 * network.layers[i-2].parameters.patch_size,
					in_fan = (overlap ? 9 : 4) * sl[i-1],
					n_of_sparse_layer_patches = (overlap ? Int((sqrt(network.layers[i-1].parameters.n_of_pool_layer_patches)-1)/2)^2 :
					Int(network.layers[i-1].parameters.n_of_pool_layer_patches/4))))
			else
				push!(network.layers,layer_sparse_patchy(sl[i-1:i]))
			end
		elseif tl[i] == "pool"
			if tl[i-1] == "sparse_patchy"
				push!(network.layers,layer_pool([network.layers[i-1].parameters.n_of_sparse_layer_patches * sl[i-1],sl[i]]))
			elseif tl[i-1] == "sparse"
				push!(network.layers,layer_pool(sl[i-1:i]))
			else
				error("Pool layer should come after sparse layer!")
			end
		elseif tl[i] == "pool_patchy"
			if tl[i-1] == "sparse_patchy"
				push!(network.layers,layer_pool_patchy(sl[i-1:i];
					n_of_pool_layer_patches = network.layers[i-1].parameters.n_of_sparse_layer_patches))
			else
				error("Pool patch layer should come after sparse patch layer!")
			end
		elseif tl[i] == "classifier"
			if tl[i-1] == "sparse_patchy"
				push!(network.layers,classifier([network.layers[i-1].parameters.n_of_sparse_layer_patches * sl[i-1],sl[i]]))
			elseif tl[i-1] == "pool_patchy"
				push!(network.layers,classifier([network.layers[i-1].parameters.n_of_pool_layer_patches * sl[i-1],sl[i]]))
			else
				push!(network.layers,classifier(sl[i-1:i]))
			end
		end
	end
	return network
end

#############################################################
#Configuration Type

#UNDER CONSTRUCTION!!!
struct config
	task::String
	layer_sizes::Array{Int64, 1}
  	n_inits::Int64
	iterations::Int64
	learningrates::Array{Float64, 1}
	lambda::Array{Float64, 1}
	weight_decay::Float64
	nonlinearity::Array{Function, 1}
	nonlinearity_diff::Array{Function, 1}
	nonlin_diff::Bool
end
