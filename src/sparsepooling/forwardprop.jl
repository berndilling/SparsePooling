
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

#for pooling layers
#ATTENTION: FOR PCA nonlinearity should be linear!
#tau_u: time constant of low-pass filter of membrane potential, measured in units of inputs/iterations/data presentations (= dt in this case)
#lin: boolian if linear (no nonlinearity, no biases) forwardprop should be executed (for PCA)
#calculate_trace: boolian if trace (low-pass filtered membr. pot.) should be calculated
function forwardprop!(layer_pre, layer_post::layer_pool; nonlinearity = lin!, one_over_tau_u = 1e-1, lin = true, calculate_trace = true)
	if lin
		BLAS.gemv!('N', 1., layer_post.w, layer_pre.a, 0., layer_post.u) # membrane potential = weighted sum over inputs
		layer_post.a = deepcopy(layer_post.u)
	else
		BLAS.gemv!('N', 1., layer_post.w, layer_pre.a, 0., layer_post.u) # membrane potential = weighted sum over inputs
		BLAS.axpy!(1., layer_post.b, layer_post.u) # add bias term
		nonlinearity(layer_post.u, layer_post.a) # apply non-linearity
		if calculate_trace
			layer_post.u_tr = (1-one_over_tau_u)*layer_post.u_tr + one_over_tau_u*layer_post.u # update low-pass filtered membrane potential
		end
	end
end

# Activation function with threshold
function _activation_function!(input,output,threshold)
	for i in 1:length(input)
		output[i] = clamp(input[i]-threshold[i],0.,Inf64) #thresholded, linear rectifier
	end
end

# Rate implementation of SC algorithm by Zylberberg et al PLoS Comp Bio 2011
# Similar to Brito's sparse coding algorithm
# time constant tau of DEQ equals: tau = 1
# dt is measured in units of: tau = 1 and it should be: dt << tau = 1
function forwardprop!(layer_pre, layer_post::layer_sparse; dt = 1e-1, epsilon = 1e-2)
	scaling_factor = epsilon/dt
	voltage_incr = scaling_factor*norm(layer_post.u)+1 #+1 to make sure loop is entered
	input_without_recurrence = BLAS.gemv('N',layer_post.w,layer_pre.a)
	while norm(voltage_incr) > scaling_factor*norm(layer_post.u)
		voltage_incr = input_without_recurrence - BLAS.gemv('N',layer_post.v,layer_post.a) - layer_post.u
		BLAS.axpy!(dt, voltage_incr, layer_post.u) # update membrane potential
		_activation_function!(layer_post.u,layer_post.a,layer_post.t) # apply thresholded linear rectifier non-linearity
	end
end
