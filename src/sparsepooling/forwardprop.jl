
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
function calculatetrace!(layer_post)
	layer_post.a_tr = (1-layer_post.parameters.one_over_tau_a)*layer_post.a_tr + layer_post.parameters.one_over_tau_a*layer_post.a
end
#for pooling layers
#ATTENTION: FOR PCA nonlinearity should be linear!
#tau_a: time constant of low-pass filter of activity, measured in units of inputs/iterations/data presentations (= dt in this case)
#lin: boolian if linear (no nonlinearity, no biases) forwardprop should be executed (for PCA)
#calculate_trace: boolian if trace (low-pass filtered activity) should be calculated
function forwardprop!(layer_pre, layer_post::layer_pool)
	if layer_post.parameters.calculate_trace
		calculatetrace!(layer_post)
		calculatetrace!(layer_pre)
	end
	if layer_post.parameters.nonlinearity == "linear"
		BLAS.gemv!('N', 1., layer_post.w, layer_pre.a, 0., layer_post.u) # membrane potential = weighted sum over inputs
		layer_post.a = deepcopy(layer_post.u)
	elseif layer_post.parameters.nonlinearity == "relu"
		BLAS.gemv!('N', 1., layer_post.w, layer_pre.a, 0., layer_post.u) # membrane potential = weighted sum over inputs
		BLAS.axpy!(1., layer_post.b, layer_post.u) # add bias term
		relu!(layer_post.u, layer_post.a) # apply non-linearity
	else
		error("non-linear option not defined yet in pooling layer!")
	end
end

# Activation function with threshold
function _activation_function!(input,output,threshold,function_type,OneOverMaxFiringRate)
	if function_type == "relu"
		for i in 1:length(input)
			output[i] = clamp(input[i]-threshold[i],0.,Inf64) #thresholded, linear rectifier
		end
	elseif function_type == "resqroot"
		for i in 1:length(input)
			output[i] = sqrt(clamp(input[i]-threshold[i],0.,Inf64)) #thresholded square root
		end
	elseif function_type == "heavyside"
		for i in 1:length(input)
			output[i] = Float64(input[i] > threshold[i])# heavyside
		end
	elseif function_type == "pwl"
		for i in 1:length(input)
			output[i] = clamp(input[i]-threshold[i],0.,1.) #piece-wise linear
		end
	elseif function_type == "LIF"
		for i in 1:length(input)
			if input[i] <= threshold[i]
				output[i] = 0.
			else
				output[i] = 1./(OneOverMaxFiringRate-log(1-threshold[i]/input[i]))
			end
		end
	end
end


# Activation function of an Leaky Integrate and Fire model (with refractatoriness)
# IMPLIES U_RESET = 0!
# USE ONLY IF THRESHOLDS (AND MEMBRANE POTENTIALS?) ARE ABOVE 0!
# OneOverMaxFiringRate: Parameter for refrect.: 0 -> no refractatoriness
# Zylberberg has 50 as maximum spike rate in his model! -> OneOverMaxFiringRate = 1/50
function _activation_function_refractLIF!(input,output,threshold; OneOverMaxFiringRate = 1/50)
	for i in 1:length(input)
		if input[i] <= threshold[i]
			output[i] = 0.
		else
			output[i] = 1./(OneOverMaxFiringRate-log(1-threshold[i]/input[i]))
		end
	end
end

# Rate implementation of SC algorithm by Zylberberg et al PLoS Comp Bio 2011
# Similar to Brito's sparse coding algorithm
# time constant tau of DEQ equals: tau = 1
# dt is measured in units of: tau = 1 and it should be: dt << tau = 1
function forwardprop!(layer_pre, layer_post::layer_sparse)
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
		_activation_function!(layer_post.u,layer_post.a,layer_post.t,
			layer_post.parameters.activationfunction,
			layer_post.parameters.OneOverMaxFiringRate) # apply activation function
	end
end
