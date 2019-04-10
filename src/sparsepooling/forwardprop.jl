
using LinearAlgebra

###############################################################################
# forwardpropagation of activity in layers

#for classifier
@inline function forwardprop!(net::classifier)
	net.a[1] = deepcopy(net.a_pre)
	for i in 1:net.nl
		BLAS.gemv!('N', 1., net.w[i], net.a[i], 0., net.u[i])
		BLAS.axpy!(1., net.b[i], net.u[i])
		net.activationfunctions[i](net.u[i], net.a[i+1])
	end
end

# update low-pass filtered activity (before updating activity since current step should not be included, see Robinson&Rolls paper)
@inline function calculatetrace!(layer)
	layer.a_tr = (1-layer.parameters.one_over_tau_a) .* layer.a_tr + layer.parameters.one_over_tau_a .* layer.a
end
@inline function calculatetrace!(layer::layer_sparse)
	layer.a_tr = (1-layer.parameters.one_over_tau_a)*layer.a_tr + layer.parameters.one_over_tau_a*layer.a
	layer.a_tr_s = (1-layer.parameters.one_over_tau_a_s)*layer.a_tr_s + layer.parameters.one_over_tau_a_s*layer.a
end


#Forwardprop WITHOUT lateral competition (wlc): meant for pooling layers!
#ATTENTION: FOR PCA/SFA nonlinearity should be linear!
#PAY ATTENTION: lc_forward has to be consistent with the one in parameterupdate!
@inline function forwardprop!(layer::layer_pool; lc_forward = true) #true
	lc_forward ? forwardprop_lc!(layer) : #forwardprop_WTA!(layer) : #
		forwardprop_wlc!(layer)
end
@inline function forwardprop_wlc!(layer)
	layer.parameters.calculate_trace &&	calculatetrace!(layer)
	#THIS IS PROBABLY WRONG SINCE IT DESTROYS TIME SCALE
	#if norm(layer.a_pre) != 0.
	BLAS.gemv!('N', 1., layer.w, layer.a_pre, 0., layer.u) # membrane potential = weighted sum over inputs
	BLAS.axpy!(1., layer.b, layer.u) # add bias term
	layer.parameters.activationfunction(layer) # apply activation function
	#end
end

# Forwardprop WITH lateral competition (lc)
# Rate implementation of SC algorithm by Zylberberg et al PLoS Comp Bio 2011
# Similar to Brito's sparse coding algorithm
# time constant tau of DEQ equals: tau = 1
# dt is measured in units of: tau = 1 and it should be: dt << tau = 1
@inline function forwardprop!(layer::layer_sparse)
	forwardprop_lc!(layer)
	#forwardprop_WTA!(layer)
end
@inline function forwardprop_lc!(layer::layer)
	#if (norm(layer.a_pre) != 0.) && (norm(layer.a) != 0.) # IS THIS BIO-PLAUSIBLE???
	if norm(layer.a) != 0.
		layer.parameters.calculate_trace &&	calculatetrace!(layer)
	end
	layer.u .= 0.
	layer.a .= 0.
	if norm(layer.a_pre) != 0.
		if layer.parameters.activationfunction == lin!
			layer.u = inv(eye(size(layer.v)[1]) .+ layer.v) * layer.w * layer.a_pre
			layer.parameters.activationfunction(layer) #linear, just to assign values from u to a
		else
			scaling_factor = layer.parameters.epsilon/layer.parameters.dt
			voltage_incr = scaling_factor*norm(layer.u)+1 #+1 to make sure loop is entered
			input_without_recurrence = BLAS.gemv('N',layer.w,layer.a_pre)
			while norm(voltage_incr) > scaling_factor * norm(layer.u) #for _ in 1:20 #
				voltage_incr = input_without_recurrence - BLAS.gemv('N',layer.v,layer.a) - layer.u
				BLAS.axpy!(layer.parameters.dt, voltage_incr, layer.u) # update membrane potential
				layer.parameters.activationfunction(layer) # apply activation function
			end
		end
	end
end
# ATTENTION: Only for testing, no explicit (plastic) lateral inhibition used!!!
# sets winner (highest input) to 1, all other to 0
@inline function forwardprop_WTA!(layer)
	if norm(layer.a) != 0.
		layer.parameters.calculate_trace &&	calculatetrace!(layer)
	end
	layer.u .= 0.
	layer.a .= 0.
	if norm(layer.a_pre) != 0.
		input_without_recurrence = BLAS.gemv('N',layer.w,layer.a_pre)
		maxinput = findmax(input_without_recurrence)
		#(maxinput[1] >= layer.t[maxinput[2]]) && (layer.a[maxinput[2]] = 1.)
		layer.a[maxinput[2]] = 1.
	end
end


@inline function forwardprop!(layer::layer_patchy; normalize = false)
	len = length(layer.layer_patches[1].a)
	i = 1
	for layer_patch in layer.layer_patches
		forwardprop!(layer_patch)
		copyto!(layer.a, (i-1)*len+1:i*len,	layer_patch.a, 1:len)
		copyto!(layer.a_tr, (i-1)*len+1:i*len, layer_patch.a_tr, 1:len)
		i += 1
	end
	if normalize # take care this only works for layer_sparse_patchy yet!
		(maximum(layer.a) > layer.a_max) && (layer.a_max = deepcopy(maximum(layer.a)))
		layer.a ./= layer.a_max; layer.a_tr ./= layer.a_max
	end
end

###############################################################################
# For whole net until specified layer

@inline function forwardprop!(net::net; FPUntilLayer = net.nr_layers)
	calculatetrace!(net.layers[1])
	for i in 1:FPUntilLayer-1
		distributeinput!(net.layers[i],net.layers[i+1])
		forwardprop!(net.layers[i+1])
	end
end


###########################################################################
# For broadcasting and distributing input on patches (patch-connectivity is realized here!)

@inline getindx1(i, j, n_rows) = (i-1) * n_rows + j
@inline getindx2(i, j, i1, j1, str, isize) = (i-1)*str+i1 + (j-1)*str*isize + (j1-1)*isize
@inline function getparameters(layer::layer_patchy)
	return Int(sqrt(layer.parameters.n_of_layer_patches)),
		layer.parameters.patch_size, layer.parameters.in_size,
		layer.parameters.stride
end
# input layer -> patchy layer
@inline function copyinput!(dest::Array{Float64, 1}, src::Array{Float64, 1},
							i::Int64, j::Int64, psize::Int64, isize::Int64, str::Int64)
	copyto!(reshape(dest, psize, psize), CartesianIndices((1:psize, 1:psize)),
			reshape(src, isize, isize),
			CartesianIndices(((i-1)*str+1:(i-1)*str+psize, (j-1)*str+1:(j-1)*str+psize)))
end
# patchy layer -> patchy layer
@inline function copyinput!(dest::Array{Float64, 1}, src::Array{Float64, 1},
							i1::Int64, j1::Int64, n_neurons_per_pop::Int64, p_size::Int64)

	#TODO: fix bug here!!!

	copyto!(dest, getindx1(i1, j1, p_size):getindx1(i1, j1, p_size)+n_neurons_per_pop-1,
			src, 1:length(src))
end
# for input layer -> patchy layer
@inline function distributeinput!(layer_pre::layer_input, layer_post::layer_sparse_patchy)
	n_patch, p_size, i_size, str = getparameters(layer_post)
	for i in 1:n_patch
		for j in 1:n_patch
			copyinput!(layer_post.layer_patches[getindx1(i, j, n_patch)].a_pre,
			 		   layer_pre.a,	i, j, p_size, i_size, str)
		   	copyinput!(layer_post.layer_patches[getindx1(i, j, n_patch)].a_tr_pre,
			 		   layer_pre.a_tr, i, j, p_size, i_size, str)
		end
	end
end
# for patchy layer -> patchy layer
@inline function distributeinput!(layer_pre::layer_patchy, layer_post::layer_patchy)
	n_patch, p_size, i_size, str = getparameters(layer_post)
	n_neurons_per_pop = length(layer_pre.layer_patches[1].a)
	for i in 1:n_patch
		for j in 1:n_patch
			for i1 in 1:p_size
				for j1 in 1:p_size
					copyinput!(layer_post.layer_patches[getindx1(i, j, n_patch)].a_pre,
					 			layer_pre.layer_patches[getindx2(i, j, i1, j1, str, i_size)].a,
								i1, j1, n_neurons_per_pop, p_size)
					copyinput!(layer_post.layer_patches[getindx1(i, j, n_patch)].a_tr_pre,
					 			layer_pre.layer_patches[getindx2(i, j, i1, j1, str, i_size)].a_tr,
								i1, j1, n_neurons_per_pop, p_size)
					# copyinput!(layer_post.layer_patches[getindx1(i, j, n_patch)].a_tr_s_pre,
					#  			layer_pre.layer_patches[getindx2(i, j, i1, j1, str, i_size)].a_tr_s,
					# 			i1, j1, n_neurons_per_pop)
				end
			end
		end
	end
end
#For all the other situations (fully connected):
@inline function distributeinput!(layer_pre, layer_post)
	inds = 1:length(layer_post.a_pre)
	copyto!(layer_post.a_pre, inds, layer_pre.a, inds)
	copyto!(layer_post.a_tr_pre, inds, layer_pre.a_tr, inds)
end
