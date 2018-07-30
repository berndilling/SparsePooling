
###############################################################################
# forwardpropagation of activity in layers

#for classifier
@inline function forwardprop!(net::classifier)
	for i in 1:net.nl
		BLAS.gemv!('N', 1., net.w[i], net.x[i], 0., net.ax[i])
		BLAS.axpy!(1., net.b[i], net.ax[i])
		net.activationfunctions[i](net.ax[i], net.x[i+1])
	end
end

# update low-pass filtered activity (before updating activity since current step should not be included, see Robinson&Rolls paper)
@inline function calculatetrace!(layer)
	layer.a_tr = (1-layer.parameters.one_over_tau_a)*layer.a_tr + layer.parameters.one_over_tau_a*layer.a
end

#Forwardprop WITHOUT lateral competition (wlc): meant for pooling layers!
#ATTENTION: FOR PCA nonlinearity should be linear!
@inline function forwardprop!(layer_pre, layer_post::layer_pool, patches::Array{Float64, 3})
	forwardprop_wlc!(layer_pre, layer_post)
end
@inline function forwardprop!(layer_pre::layer_input, layer_post::layer_pool, image::Array{Float64, 2})
	layer_pre.a = image[:]
	forwardprop_wlc!(layer_pre, layer_post)
end
@inline function forwardprop_wlc!(layer_pre, layer_post)
	if typeof(layer_pre) == layer_sparse_patchy
		calculatetrace!(layer_post) #pre trace is already calculated
	else
		if layer_post.parameters.calculate_trace
			calculatetrace!(layer_post)
			calculatetrace!(layer_pre)
		end
	end
	BLAS.gemv!('N', 1., layer_post.w, layer_pre.a, 0., layer_post.u) # membrane potential = weighted sum over inputs
	BLAS.axpy!(1., layer_post.b, layer_post.u) # add bias term
	layer_post.parameters.activationfunction(layer_post) # apply activation function
end

# Forwardprop WITH lateral competition (lc)
# Rate implementation of SC algorithm by Zylberberg et al PLoS Comp Bio 2011
# Similar to Brito's sparse coding algorithm
# time constant tau of DEQ equals: tau = 1
# dt is measured in units of: tau = 1 and it should be: dt << tau = 1
@inline function forwardprop!(layer_pre::layer_pool, layer_post::layer_sparse, patches::Array{Float64, 3})
	forwardprop_lc!(layer_pre, layer_post)
end
@inline function forwardprop!(layer_pre::layer_input, layer_post::layer_sparse)
	forwardprop_lc!(layer_pre, layer_post)
end
@inline function forwardprop!(layer_pre::layer_input, layer_post::layer_sparse, image::Array{Float64, 2})
	layer_pre.a = image[:]
	forwardprop_lc!(layer_pre, layer_post)
end
@inline function forwardprop_lc!(layer_pre, layer_post)
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
		layer_post.parameters.activationfunction(layer_post) # apply activation function
	end
end

@inline function forwardprop!(layer_pre::layer_input, layer_post::layer_sparse_patchy, patches::Array{Float64, 3})
	layer_post.a, layer_post.a_tr = [], []
	i = 1
	for sparse_layer_patch in layer_post.sparse_layer_patches
		if norm(patches[:,:,i]) != 0
			layer_pre.a = patches[:,:,i][:]
			sparse_layer_patch.u = zeros(length(sparse_layer_patch.u))
			sparse_layer_patch.a = zeros(length(sparse_layer_patch.a))
			forwardprop!(layer_pre, sparse_layer_patch)
			append!(layer_post.a, sparse_layer_patch.a)
			append!(layer_post.a_tr, sparse_layer_patch.a_tr)
		else
			append!(layer_post.a,zeros(length(sparse_layer_patch.a)))
			append!(layer_post.a_tr,zeros(length(sparse_layer_patch.a_tr)))
		end
		i += 1
	end
end

###############################################################################
# For whole net until specified layer

@inline function forwardprop!(net::net; FPUntilLayer = net.nr_layers - 1)
	for i in 1:FPUntilLayer-1
		distributeinput!(net.layers[i],net.layers[i+1])
		forwardprop!(net.layers[i+1])
	end
end


###########################################################################
# For broadcasting and distributing input on patches (patch-connectivity is realized here!)
# Mainly used for (potential parallelization)
# Not needed for fully connected layers!

@inline function distributeinput!(layer_pre::layer_input, layer_post::layer_sparse_patchy)
	n_patch = Int(sqrt(layer_post.parameters.n_of_sparse_layer_patches))
	p_size = layer_post.parameters.patch_size
	i_size = layer_post.parameters.image_size
	ol = layer_post.parameters.overlap
	# TODO optimize and parallelize this!!!
	for i in 1:n_patch
		for j in 1:n_patch
			layer_post.sparse_layer_patches[(i-1)*n_patch+j].a_pre =
				reshape(layer_pre.a,i_size,i_size)[(i-1)*(p_size-ol)+1:i*(p_size)-(i-1)*ol,
              (j-1)*(p_size-ol)+1:j*(p_size)-(j-1)*ol]
			layer_post.sparse_layer_patches[(i-1)*number_of_patches+j].a_tr_pre =
				reshape(layer_pre.a_tr,i_size,i_size)[(i-1)*(p_size-ol)+1:i*(p_size)-(i-1)*ol,
              (j-1)*(p_size-ol)+1:j*(p_size)-(j-1)*ol]
		end
	end
end
@inline function distributeinput!(layer_pre::layer_sparse_patchy, layer_post::layer_pool_patchy)
	(layer_pre.parameters.n_of_sparse_layer_patches != layer_post.parameters.n_of_pool_layer_patches) &&
		error("pool patches must be same number as sparse patches in previous layer!")
	for i in 1:layer_post.parameters.n_of_pool_layer_patches
		layer_post.pool_layer_patches[i].a_pre = deepcopy(layer_post.sparse_layer_patches[i].a)
		layer_post.pool_layer_patches[i].a_tr_pre = deepcopy(layer_post.sparse_layer_patches[i].a_tr)
	end
end
@inline function distributeinput!(layer_pre::layer_pool_patchy, layer_post::layer_sparse_patchy)
end
@inline function distributeinput!(layer_pre::layer_pool_patchy, layer_post::layer_sparse)
end
@inline function distributeinput!(layer_pre::layer_sparse, layer_post::layer_pool)
end
@inline function distributeinput!(layer_pre::layer_pool, layer_post::classifier)
end
