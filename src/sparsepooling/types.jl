
#############################################################
#Basic types and constructors

## LAYERS

type layer_input
	a::Array{Float64, 1} #activation
end

type layer_sparse
	u::Array{Float64, 1} #membrane potential
	a::Array{Float64, 1} #activation = nonlinearity(membrane potential)
	w::Array{Float64, 2} #synaptic weight matrix in format TOxFROM
	v::Array{Float64, 2} #recurrent/lateral inhibition weight matrix
	t::Array{Float64, 1} #thresholds
	p::Float64 #average activation/"firing rate"
	hidden_reps::Array{Float64, 2} #hidden representations of the layer (saved to accelerate further learning)
end

type layer_pool
	u::Array{Float64, 1} #membrane potential
	a::Array{Float64, 1} #activation = nonlinearity(membrane potential)
	a_tr::Array{Float64, 1} #low pass filtered activity: "trace" (current time step is left out)
	w::Array{Float64, 2} #synaptic weight matrix in format TOxFROM
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
	layer_types::Array{String, 1} #type of layers: sparse or pool (...maybe others later)
	layers::Array{Any, 1} #layers of the network
end

############################################################################
## CONSTRUCTORS

function layer_input(ns::Int64) #ns: number of neurons in input layer
	layer_input(zeros(ns))
end

function layer_sparse(ns::Array{Int64, 1}) #ns: number of neurons in previous and present layer
	layer_sparse(zeros(ns[2]), #membrane potential initialized with zeros
			zeros(ns[2]), #activation initialized with zeros
			randn(ns[2], ns[1])/(10*sqrt(ns[1])), #feed-forward weights initialized gaussian distr.
			zeros(ns[2], ns[2]), #lateral inhibition initialized with zeros
			5*ones(ns[2]), #thresholds initialized with 5's (as in Zylberberg) (zero maybe not so smart...)
			0.05, #average activation set to 5% (as in Zylberberg)
			zeros(ns[2],10)) #reps initialized with zeros (only 10 reps here, but can be changed later)
end

function layer_pool(ns::Array{Int64, 1})
	layer_pool(zeros(ns[2]), #membrane potential initialized with zeros
			zeros(ns[2]), #low-pass filtered membrane potential initialized with zeros
			zeros(ns[2]), #activation initialized with zeros
			randn(ns[2], ns[1])/(10*sqrt(ns[1])), #feed-forward weights initialized gaussian distr.
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
