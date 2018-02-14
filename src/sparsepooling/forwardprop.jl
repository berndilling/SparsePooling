
#####################################################
#forwardpropagation of activity

#for classifier for different nonlinearities
function forwardprop!(net; nonlinearity = Array{Function, 1})
	for i in 1:net.nl
		BLAS.gemv!('N', 1., net.w[i], net.x[i], 0., net.ax[i])
		BLAS.axpy!(1., net.b[i], net.ax[i])
		nonlinearity[i](net.ax[i], net.x[i+1])
	end
end

# update low-pass filtered activity (before updating activity since current step should not be included, see Robinson&Rolls paper)
function calculatetrace!(layer)
	layer.a_tr = (1-layer.parameters.one_over_tau_a)*layer.a_tr + layer.parameters.one_over_tau_a*layer.a
end

# Activation functions
function _activation_function!(layer)
	if layer.parameters.activationfunction == "linear"
		for i in 1:length(layer.u)
			layer.a[i] = deepcopy(layer.u[i])
		end
	elseif layer.parameters.activationfunction == "relu"
		for i in 1:length(layer.u)
			layer.a[i] = clamp(layer.u[i]-layer.t[i],0.,Inf64) #thresholded, linear rectifier
		end
	elseif layer.parameters.activationfunction == "resqroot"
		for i in 1:length(layer.u)
			layer.a[i] = sqrt(clamp(layer.u[i]-layer.t[i],0.,Inf64)) #thresholded square root
		end
	elseif layer.parameters.activationfunction == "heavyside"
		for i in 1:length(layer.u)
			layer.a[i] = Float64(layer.u[i] > layer.t[i])# heavyside
		end
	elseif layer.parameters.activationfunction == "pwl"
		for i in 1:length(layer.u)
			layer.a[i] = clamp(layer.u[i]-layer.t[i],0.,1.) #piece-wise linear
		end
	elseif layer.parameters.activationfunction == "sigm"
		for i in 1:length(layer.u)
			layer.a[i] = 1./(1.+exp(-layer.u[i]))
		end
	elseif layer.parameters.activationfunction == "LIF"
		# Activation function of an Leaky Integrate and Fire model (with refractatoriness)
		# IMPLIES U_RESET = 0!
		# USE ONLY IF THRESHOLDS (AND MEMBRANE POTENTIALS?) ARE ABOVE 0!
		# OneOverMaxFiringRate: Parameter for refrect.: 0 -> no refractatoriness
		# Zylberberg has 50 as maximum spike rate in his model! -> OneOverMaxFiringRate = 1/50
		for i in 1:length(layer.u)
			if layer.u[i] <= layer.t[i]
				layer.a[i] = 0.
			else
				layer.a[i] = 1./(layer.parameters.OneOverMaxFiringRate-log(1-layer.t[i]/layer.u[i]))
			end
		end
	end
end


#Forwardprop WITHOUT lateral competition (wlc): meant for pooling layers!
#ATTENTION: FOR PCA nonlinearity should be linear!
#tau_a: time constant of low-pass filter of activity, measured in units of inputs/iterations/data presentations (= dt in this case)
#lin: boolian if linear (no nonlinearity, no biases) forwardprop should be executed (for PCA)
#calculate_trace: boolian if trace (low-pass filtered activity) should be calculated
function forwardprop!(layer_pre, layer_post::layer_pool)
	forwardprop_wlc!(layer_pre, layer_post)
end
function forwardprop_wlc!(layer_pre, layer_post)
	if layer_post.parameters.calculate_trace
		calculatetrace!(layer_post)
		calculatetrace!(layer_pre)
	end
	BLAS.gemv!('N', 1., layer_post.w, layer_pre.a, 0., layer_post.u) # membrane potential = weighted sum over inputs
	BLAS.axpy!(1., layer_post.b, layer_post.u) # add bias term
	_activation_function!(layer_post) # apply activation function
end

# Forwardprop WITH lateral competition (lc)
# Rate implementation of SC algorithm by Zylberberg et al PLoS Comp Bio 2011
# Similar to Brito's sparse coding algorithm
# time constant tau of DEQ equals: tau = 1
# dt is measured in units of: tau = 1 and it should be: dt << tau = 1
function forwardprop!(layer_pre, layer_post::layer_sparse)
	forwardprop_lc!(layer_pre, layer_post)
end
function forwardprop_lc!(layer_pre, layer_post)
	if layer_post.parameters.calculate_trace
		calculatetrace!(layer_post)
		calculatetrace!(layer_pre)
	end
	scaling_factor = layer_post.parameters.epsilon/layer_post.parameters.dt
	voltage_incr = scaling_factor*norm(layer_post.u)+1 #+1 to make sure loop is entered
	input_without_recurrence = BLAS.gemv('N',layer_post.w,layer_pre.a)
	while norm(voltage_incr) > scaling_factor*norm(layer_post.u)
		voltage_incr = input_without_recurrence - BLAS.gemv('N',layer_post.v,layer_post.a) - layer_post.u
		BLAS.axpy!(layer_post.parameters.dt, voltage_incr, layer_post.u) # update membrane potential
		_activation_function!(layer_post) # apply activation function
	end
end

function forwardprop!(layer_pre, layer_post::layer_sparse_patchy, patches::Array{Float64, 3})
	layer_post.common_a, layer_post.common_a_tr = [], []
	for i in 1:layer_post.parameters.n_of_sparse_layer_patches
		if norm(patches[:,:,i]) != 0
			layer_pre.a = patches[:,:,i][:]
			forwardprop_lc!(layer_pre, layer_post.sparse_layer_patches[i])
			append!(layer_post.common_a,layer_post.sparse_layer_patches[i].a)
			append!(layer_post.common_a_tr,layer_post.sparse_layer_patches[i].a_tr)
		else
			append!(layer_post.common_a,zeros(length(layer_post.sparse_layer_patches[i].a)))
			append!(layer_post.common_a_tr,zeros(length(layer_post.sparse_layer_patches[i].a_tr)))
		end
	end
end
