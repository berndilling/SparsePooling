
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
function forwardprop!(layer_pre, layer_post::layer_pool, nonlinearity = Function)
	BLAS.gemv!('N', 1., layer_post.w, layer_pre.a, 0., layer_post.u)
	BLAS.axpy!(1., layer_post.b, layer_post.u)
	nonlinearity(layer_post.u, layer_post.a)
end

# Activation function with threshold
function activation_function!(input,output,lambda)
	for i in 1:length(input)
		output[i] = clamp(input[i]-lambda,0.,Inf64) #thresholded, linear rectifier
	end
end

# Rate implementation of SC algorithm by Zylberberg et al PLoS Comp Bio 2011
# Similar to Brito's sparse coding algorithm
# time constant tau of DEQ equals: tau = 1
# dt is measured in units of: tau = 1
function forwardprop!(payer_pre, layer_post::layer_sparse; iterations = 50, dt = 1e-1)
	for i in 1:iterations #or until convergence...
		layer_post.u = dt*(layer_post.w*layer_pre.a - layer_post.v*layer_post.a)+(1-dt)*layer_post.u
		activation_function(layer_post.u,layer_post.a)
	end
end

#IMPLEMENT RECURRENCE ETC.!!!


#Britos algorithm
function forwardprop!(layer_pre,layer_post::layer_sparse, iter::Int64; iterations = 50, r = 1e-1,	learningrate_inh = 1e-2)
	if iter == -1
		one_over_iter = 0
	else
		one_over_iter = 1./convert(Float64, iter)
	end
	for i in 1:net.nl
		if nonlinearity[i] != lin!
			error("nonlinearity != linear. Must be linear for sparse coding!")
		else
			net.x[i+1][:] = 0.
			net.ax[i][:] = 0.
			n,m = size(net.w[i])
			for j in 1:n
				net.A[i][j,j] = 0. #no self-inhibition
			end
			input = BLAS.gemv('N',net.w[i],net.x[i])
			for k in 1:iterations
				net.ax[i] = r*(input-BLAS.gemv('N',net.A[i],net.x[i+1]))+(1.0-r)*net.ax[i]
				activation_Brito!(net.ax[i],net.x[i+1],lambdas[i])
			end
			if iter != -1 #only learn lateral inhibition weights during overall learning procedure, not during generatehiddenreps!
				if one_over_iter > memory_decay #avoids zero-filling at beginning
					net.B[i][1,:] = (1.-one_over_iter)*net.B[i][1,:] + one_over_iter*net.x[i+1]
				else
					net.B[i][1,:] = (1.-memory_decay)*net.B[i][1,:] + memory_decay*net.x[i+1]
				end
				BLAS.ger!(learningrate_inh,net.x[i+1]-net.B[i][1,:],net.x[i+1],net.A[i])
				clamp!(net.A[i],0.,Inf64) #Dale's law
			end
		end
	end
end
