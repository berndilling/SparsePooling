
#############################################################
#Basic types and constructors

## LAYERS
type parameters_input
	one_over_tau_a::Float64
end
function parameters_input(; one_over_tau_a = 1e-3)
	parameters_input(one_over_tau_a)
end
type layer_input
	parameters::parameters_input
	a::Array{Float64, 1} #activation
	a_tr::Array{Float64, 1} #low pass filtered activity: "trace" (current time step is left out)
end

type parameters_sparse
	learningrate_v::Float64 # learning rate for lateral inhibition
	learningrate_w::Float64 # learning rate for feedforward weights
	learningrate_thr::Float64 # learning rate for thresholds
	dt::Float64 #time step for integration
	epsilon::Float64 #convergence criterium for recurrence
	activationfunction::String #"relu"/"resqroot"/"heavyside"/"pwl"/"sigm"/"LIF"
	OneOverMaxFiringRate::Float64 #used for LIF nonlin. determines refractory period/saturation
	calculate_trace::Bool
	one_over_tau_a::Float64 # time constant for low pass filtered activity "trace"
	p::Float64 #average activation/"firing rate"
end
function parameters_sparse(; learningrate_v = 1e-1, learningrate_w = 1e-3, learningrate_thr = 1e-2,
		dt = 1e-1, epsilon = 1e-4, activationfunction = "pwl", OneOverMaxFiringRate = 1/50,
		calculate_trace = true, one_over_tau_a = 1e-1, p = 1/12) #p: average activation set to 5% (as in Zylberberg)
	parameters_sparse(learningrate_v, learningrate_w, learningrate_thr,
			dt, epsilon, activationfunction, OneOverMaxFiringRate,
			calculate_trace, one_over_tau_a, p)
end
type layer_sparse
	parameters::parameters_sparse
	u::Array{Float64, 1} #membrane potential
	a::Array{Float64, 1} #activation = nonlinearity(membrane potential)
	a_tr::Array{Float64, 1} #low pass filtered activity: "trace" (current time step is left out)
	w::Array{Float64, 2} #synaptic weight matrix in format TOxFROM
	v::Array{Float64, 2} #recurrent/lateral inhibition weight matrix
	t::Array{Float64, 1} #thresholds
	hidden_reps::Array{Float64, 2} #hidden representations of the layer (saved to accelerate further learning)
end

type parameters_sparse_patchy
	n_of_sparse_layer_patches::Int64 #number of independent sparse layer patches
	patch_size::Int64 #number of each in-fan: patch_size*patch_size
	overlap::Int64 #overlap of patches in pixels
end
# function parameters_sparse_patchy(; n_of_sparse_layer_patches = 49,
# 	patch_size = 8, overlap = 4)
# 	parameters_sparse_patchy(n_of_sparse_layer_patches, patch_size, overlap)
# end
type layer_sparse_patchy
	parameters::parameters_sparse_patchy
	sparse_layer_patches::Array{layer_sparse, 1}
	common_a::Array{Float64, 1} #combined sparse activity of all sparse layer patches
	common_a_tr::Array{Float64, 1} #same for activity trace
end

type parameters_pool
	learningrate::Float64
	learningrate_v::Float64 # learning rate for lateral inhibition
	learningrate_w::Float64 # learning rate for feedforward weights
	learningrate_thr::Float64 # learning rate for thresholds
	dt::Float64 #time step for integration
	epsilon::Float64 #convergence criterium for recurrence
	updatetype::String
	updaterule::String #"Sanger"/"Oja"
	activationfunction::String
	calculate_trace::Bool
	one_over_tau_a ::Float64
	p::Float64
end
function parameters_pool(; learningrate = 1e-2, learningrate_v = 1e-1, learningrate_w = 1e-3, learningrate_thr = 1e-2,
		dt = 1e-1, epsilon = 1e-4, updatetype = "SFA", updaterule = "Sanger",
	activationfunction = "linear", calculate_trace = true, one_over_tau_a = 1e-1, p = 1.)
	parameters_pool(learningrate, learningrate_v, learningrate_w, learningrate_thr,
			dt, epsilon, updatetype, updaterule, activationfunction, calculate_trace, one_over_tau_a, p)
end
type layer_pool
	parameters::parameters_pool
	u::Array{Float64, 1} #membrane potential
	a::Array{Float64, 1} #activation = nonlinearity(membrane potential)
	a_tr::Array{Float64, 1} #low pass filtered activity: "trace" (current time step is left out)
	w::Array{Float64, 2} #synaptic weight matrix in format TOxFROM
	v::Array{Float64, 2} #recurrent/lateral inhibition weight matrix
	t::Array{Float64, 1} #thresholds
	b::Array{Float64, 1} #biases
	hidden_reps::Array{Float64, 2} #hidden representations of the layer
end

# Supervised classifier on the activations of a "net" (could access multiple levels of hierarchy!)
type classifier
	nl::Int64 #number of layers in classifier (without input layer which is part of the net)
	u::Array{Array{Float64, 1}, 1} #membrane potential
	a::Array{Array{Float64, 1}, 1} #activation = nonlinearity(membrane potential)
	e::Array{Array{Float64, 1}, 1} #error with respect to target
	w::Array{Array{Float64, 2}, 1} #synaptic weight matrix
	b::Array{Array{Float64, 1}, 1} #biases
end

## NETWORKS

#Network containing sparse and pooling layers
type net
	nr_layers::Int64 #number of layers
	layer_sizes::Array{Int64, 1} #sizes of layers (number of neurons)
	layer_types::Array{String, 1} #type of layers: sparse, sparse_patchy or pool (...maybe others later)
	layers::Array{Any, 1} #layers of the network
end

############################################################################
## CONSTRUCTORS

function layer_input(ns::Int64) #ns: number of neurons in input layer
	layer_input(parameters_input(), # default parameter init
	zeros(ns),
	zeros(ns))
end

function layer_sparse(ns::Array{Int64, 1}) #ns: number of neurons in previous and present layer
	layer_sparse(parameters_sparse(), # default parameter init
			zeros(ns[2]), #membrane potential initialized with zeros
			zeros(ns[2]), #activation initialized with zeros
			zeros(ns[2]), #low-pass filtered activity initialized with zeros
			randn(ns[2], ns[1])/(10*sqrt(ns[1])), #feed-forward weights initialized gaussian distr.
			zeros(ns[2], ns[2]), #lateral inhibition initialized with zeros
			5*ones(ns[2]), #thresholds initialized with 5's (as in Zylberberg) (zero maybe not so smart...)
			zeros(ns[2],10)) #reps initialized with zeros (only 10 reps here, but can be changed later)
end

function layer_sparse_patchy(ns::Array{Int64, 1}; n_of_sparse_layer_patches = 49,
	patch_size = 8, overlap = 4) #ns: size of in-fan and hidden layer per sparse layer patch
	layer_sparse_patchy(parameters_sparse_patchy(n_of_sparse_layer_patches, patch_size, overlap),
	[layer_sparse(ns) for i in 1:n_of_sparse_layer_patches],
	zeros(ns[2]*n_of_sparse_layer_patches),
	zeros(ns[2]*n_of_sparse_layer_patches)
	)

end

function layer_pool(ns::Array{Int64, 1})
	layer_pool(parameters_pool(), # default parameter init
			zeros(ns[2]), #membrane potential initialized with zeros
			zeros(ns[2]), #low-pass filtered activity initialized with zeros
			zeros(ns[2]), #activation initialized with zeros
			randn(ns[2], ns[1])/(10*sqrt(ns[1])), #feed-forward weights initialized gaussian distr. # rand(ns[2], ns[1])/(10*sqrt(ns[1])),#
			zeros(ns[2], ns[2]), #lateral inhibition initialized with zeros
			zeros(ns[2]), #thresholds initialized with zeros (not used up to now!)
			zeros(ns[2]), # biases equal zero for linear computation such as PCA! OR rand(ns[2])/10) #biases initialized equally distr.
			zeros(ns[2],10)) #reps initialized with zeros (only 10 reps here, but can be changed later)
end

function classifier(ns::Array{Int64, 1}) #ns: array of layer sizes in classifier
	nl = length(ns)
	classifier(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[rand(ns[i])/10 for i in 2:nl])
end

function net(sl::Array{Int64, 1}, tl::Array{String, 1}) #sl: Sizes of layers, tl: types of layers
	nl = length(sl)
	network = net(nl,sl,tl,[layer_input(sl[1])])
	for i in 2:nl
		if tl[i] == "sparse"
			push!(network.layers,layer_sparse(sl[i-1:i]))
		elseif tl[i] == "sparse_patchy"
			push!(network.layers,layer_sparse_patchy(sl[i-1:i]))
		elseif tl[i] == "pool"
			push!(network.layers,layer_pool(sl[i-1:i]))
		end
	end
	return network
end

#############################################################
#Configuration Type

#UNDER CONSTRUCTION!!!
type config
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
