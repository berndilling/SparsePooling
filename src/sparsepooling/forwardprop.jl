
#####################################################
#forwardprop for different nonlinearities

function forwardprop!(net; nonlinearity = Array{Function, 1})
	for i in 1:net.nl
		BLAS.gemv!('N', 1., net.w[i], net.x[i], 0., net.ax[i])
		BLAS.axpy!(1., net.b[i], net.ax[i])
		nonlinearity[i](net.ax[i], net.x[i+1])
	end
end


#see "g"-function in PhD-Thesis of C. Brito or Rozell2008
function activation_Brito!(input,output,lambda)
	for i in 1:length(input)
		#output[i] = (abs(input[i]) > lambda)*(input[i]-sign(input[i])*lambda)
		#output[i] = clamp((abs(input[i]) > lambda)*(input[i]-sign(input[i])*lambda),0.,Inf64)
		output[i] = clamp(input[i]-lambda,0.,Inf64) #thresholded, linear rectifier
	end
end


#full implementation with plastic lateral inhibitory connections learned by anti-Hebbian rule
#see Brito's Thesis Chapter 2
#abusing a bit the type AE_sparse by
# - saving/integrating past activations of hidden layer in B-array
# - using A-array for lateral inhibitory connections
function _forwardprop_Brito_full!(net::AE_sparse, lambdas::Array{Float64, 1}, iter::Int64;
										nonlinearity = Array{Function, 1}, iterations = 50, r = 1e-1, memory_decay = 1e-2,
										learningrate_inh = 1e-2)
										#This had worked for sure: iterations = 50, r = 1e-1, memory_decay = 1e-2,
										# 						learningrate_inh = 1e-2
										#with learningrate 1e-2
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

function forwardprop_Brito_full!(net::AE_sparse, lambdas::Array{Float64, 1}, iter::Int64;
																nonlinearity = Array{Function, 1})
	_forwardprop_Brito_full!(net, lambdas, iter; nonlinearity = nonlinearity)
end

#same for generatehiddenreps (not needed to pay attention on zeros in activities at beginning of training:
#iteration number isn't important)
function forwardprop_Brito_full!(net::AE_sparse, lambdas::Array{Float64, 1};
																nonlinearity = Array{Function, 1})
	iter = -1 #codes for iteration number isn't important
	_forwardprop_Brito_full!(net, lambdas, iter; nonlinearity = nonlinearity)
end
