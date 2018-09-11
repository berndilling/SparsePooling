
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
#ATTENTION: FOR PCA/SFA nonlinearity should be linear!
#PAY ATTENTION: lc_forward has to be consistent with the one in parameterupdate!
@inline function forwardprop!(layer::layer_pool; lc_forward = true) #true
	lc_forward ? forwardprop_lc!(layer) : forwardprop_wlc!(layer)
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
end
@inline function forwardprop_lc!(layer)
	if (norm(layer.a_pre) != 0.) && (norm(layer.a) != 0.) # IS THIS BIO-PLAUSIBLE???
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
			while norm(voltage_incr) > scaling_factor*norm(layer.u)
				voltage_incr = input_without_recurrence - BLAS.gemv('N',layer.v,layer.a) - layer.u
				BLAS.axpy!(layer.parameters.dt, voltage_incr, layer.u) # update membrane potential
				layer.parameters.activationfunction(layer) # apply activation function
			end
		end
	end
end


@inline function forwardprop!(layer::layer_sparse_patchy)
		layer.a, layer.a_tr = [], [] # combined act. of all patches
		# nthreads = Threads.nthreads()
		# N = div(layer.parameters.n_of_sparse_layer_patches, nthreads)
		# Threads.@threads for i in 1:nthreads
		# 	if i == nthreads
		# 			range = (i-1)*N + 1:layer.parameters.n_of_sparse_layer_patches
		# 	else
		# 			range = (i-1)*N + 1:i * N
		# 	end
		#	  for sparse_layer_patch in layer.sparse_layer_patches[range]
		#@sync Threads.@threads
		#@sync @parallel
		for sparse_layer_patch in layer.sparse_layer_patches
			forwardprop!(sparse_layer_patch)
			append!(layer.a, sparse_layer_patch.a)
			append!(layer.a_tr, sparse_layer_patch.a_tr)
		end
	#end
end
@inline function forwardprop!(layer::layer_pool_patchy)
	layer.a, layer.a_tr = [], [] # combined act. of all patches
	#@sync Threads.@threads
	for pool_layer_patch in layer.pool_layer_patches
		forwardprop!(pool_layer_patch)
		append!(layer.a, pool_layer_patch.a)
		append!(layer.a_tr, pool_layer_patch.a_tr)
	end
end
###############################################################################
# For whole net until specified layer

@inline function forwardprop!(net::net; FPUntilLayer = net.nr_layers - 1)
	calculatetrace!(net.layers[1])
	for i in 1:FPUntilLayer-1
		distributeinput!(net.layers[i],net.layers[i+1])
		forwardprop!(net.layers[i+1])
	end
end


###########################################################################
# For broadcasting and distributing input on patches (patch-connectivity is realized here!)
# Mainly used for (potential parallelization)
# Not needed for fully connected layers!

@inline function distributeinput!(layer_pre::layer_input, layer_post::layer_sparse)
	layer_post.a_pre = deepcopy(layer_pre.a)
	layer_post.a_tr_pre = deepcopy(layer_pre.a_tr)
end

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
              (j-1)*(p_size-ol)+1:j*(p_size)-(j-1)*ol][:]
			layer_post.sparse_layer_patches[(i-1)*n_patch+j].a_tr_pre =
				reshape(layer_pre.a_tr,i_size,i_size)[(i-1)*(p_size-ol)+1:i*(p_size)-(i-1)*ol,
              (j-1)*(p_size-ol)+1:j*(p_size)-(j-1)*ol][:]
		end
	end
end
@inline function distributeinput!(layer_pre::layer_sparse_patchy, layer_post::layer_pool_patchy)
	(layer_pre.parameters.n_of_sparse_layer_patches != layer_post.parameters.n_of_pool_layer_patches) &&
		error("pool patches must be same number as sparse patches in previous layer!")
	for i in 1:layer_post.parameters.n_of_pool_layer_patches
		layer_post.pool_layer_patches[i].a_pre = deepcopy(layer_pre.sparse_layer_patches[i].a)
		layer_post.pool_layer_patches[i].a_tr_pre = deepcopy(layer_pre.sparse_layer_patches[i].a_tr)
	end
end
@inline function distributeinput!(layer_pre::layer_pool_patchy, layer_post::layer_sparse_patchy)
	#TAKE CARE: special case of subsamplingfactor = 2 and overlap = half patchsize!!!
	n_patch_pre = Int(sqrt(layer_pre.parameters.n_of_pool_layer_patches))
	n_patch_post = Int(sqrt(layer_post.parameters.n_of_sparse_layer_patches))
	for i in 1:n_patch_post
		for j in 1:n_patch_post
			pre_a = []
			pre_a_tr = []
			for i1 in 2*i-1:2*i+1
				for j1 in 2*j-1:2*j+1
					append!(pre_a, layer_pre.pool_layer_patches[(i1-1)*n_patch_pre+j1].a)
					append!(pre_a_tr, layer_pre.pool_layer_patches[(i1-1)*n_patch_pre+j1].a_tr)
				end
			end
			layer_post.sparse_layer_patches[(i-1)*n_patch_post+j].a_pre = pre_a
			layer_post.sparse_layer_patches[(i-1)*n_patch_post+j].a_tr_pre = pre_a_tr
		end
	end
end
@inline function distributeinput!(layer_pre::layer_pool_patchy, layer_post::layer_sparse)
	layer_post.a_pre = deepcopy(layer_pre.a)
	layer_post.a_tr_pre = deepcopy(layer_pre.a_tr)
end
@inline function distributeinput!(layer_pre::layer_sparse, layer_post::layer_pool)
	layer_post.a_pre = deepcopy(layer_pre.a)
	layer_post.a_tr_pre = deepcopy(layer_pre.a_tr)
end
@inline function distributeinput!(layer_pre::layer_pool, layer_post::classifier)
	# TODO!!!
end
