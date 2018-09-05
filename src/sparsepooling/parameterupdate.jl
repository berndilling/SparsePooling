
#####################################################
#Weight/Parameter update functions

@inline function _normalize_inputweights!(weights)
	for j in 1:size(weights)[1]
		weights[j,:] *= 1./norm(weights[j,:])
	end
end

############################################################################
## Sparse layer
############################################################################

# Update rule for sparse coding algorithm: meant for both, sparse and pooling layers
#Algorithm for parameter update in sparse coding as proposed by Zylberberg et al PLoS Comp Bio 2011
# Parameters in this paper (lr_v,lr_w,lr_thr)=(0.1,0.001,0.01) (started with higher rates and then decreased)
@inline function update_layer_parameters!(layer::layer_sparse)
	 update_layer_parameters_lc!(layer)
end
@inline function update_layer_parameters_lc!(layer::layer_sparse)
	if norm(layer.a_pre) != 0. #don't do anything if no input is provided (otherwise thresholds are off)
		#Update lateral inhibition matrix
		#layer.v += lr_v*(layer.a*layer.a'-layer.parameters.p^2)
		BLAS.ger!(layer.parameters.learningrate_v,layer.a,layer.a,layer.v)
		layer.v += -layer.parameters.learningrate_v*layer.parameters.p^2
		for j in 1:size(layer.v)[1]
			layer.v[j,j] = 0. #no self-inhibition
		end
		clamp!(layer.v,0.,Inf64) #Dale's law

		#Update input weight matrix
		#Learning rule:
		#layer.w += lr_w*layer.a*(layer.a_pre-layer.a*W) # with weight decay... or explicit weight normalization/homeostasis
		#Optimized:
		# First: second term of weight update: weight decay with OLD WEIGHTS à la Oja which comes out of learning rule
		scale!((1-layer.parameters.learningrate_w*layer.a.^2),layer.w)
		# Second: First term (data-driven) of weight update
		BLAS.ger!(layer.parameters.learningrate_w,layer.a,layer.a_pre,layer.w)

		#_normalize_inputweights!(layer.w) # explicit weight normalization/homeostasis

		#Update thresholds
		#layer.t += lr_thr*(layer.a-layer.parameters.p)
		BLAS.axpy!(layer.parameters.learningrate_thr,layer.a-layer.parameters.p,layer.t)
		#avoid negative thesholds:
		#clamp!(layer.t,0.,Inf64)
	end
end
@inline function update_layer_parameters_lc!(layer::layer_pool)
	#if norm(layer.a_pre) != 0. #don't do anything if no input is provided
		#TODO Weight decay needed here? -> Yes if nonlinearity is nonlinear
		#scale!((1-layer.parameters.learningrate_w),layer.w)
		BLAS.ger!(layer.parameters.learningrate_v,layer.a_tr,layer.a_tr,layer.v)
		layer.v += -layer.parameters.learningrate_v*layer.parameters.p^2
		for j in 1:size(layer.v)[1]
			layer.v[j,j] = 0. #no self-inhibition
		end
		#TODO clamping needed here?
		clamp!(layer.v,0.,Inf64) #Dale's law
		scale!((1-layer.parameters.learningrate_w*layer.a_tr.^2),layer.w)
		#scale!((1-layer.parameters.learningrate_w),layer.w)
		#BLAS.ger!(layer.parameters.learningrate_w,layer.a_tr,layer.a_pre,layer.w)
		BLAS.ger!(layer.parameters.learningrate_w,layer.a_tr,layer.a_pre-layer.a_tr_pre,layer.w)

		#TODO threshold adaptation here? -> Yes if nonlinearity is nonlinear
		BLAS.axpy!(layer.parameters.learningrate_thr,layer.a_tr-layer.parameters.p,layer.t)
		#TODO Pre/Post-trace subtraction here?
	#end
end

@inline function update_layer_parameters!(layer::layer_sparse_patchy)
#	nthreads = Threads.nthreads()
#	N = div(layer.parameters.n_of_sparse_layer_patches, nthreads)
	# Threads.@threads for i in 1:nthreads
	# 	if i == nthreads
	# 			range = (i-1)*N + 1:layer.parameters.n_of_sparse_layer_patches
	# 	else
	# 			range = (i-1)*N + 1:i * N
	# 	end
	#		for sparse_layer_patch in layer.sparse_layer_patches[range]
		#@sync Threads.@threads
		#@sync @parallel
		for sparse_layer_patch in layer.sparse_layer_patches#[range]
		 	update_layer_parameters!(sparse_layer_patch)
		end
	#end
end
@inline function update_layer_parameters!(layer::layer_pool_patchy)
	#@sync Threads.@threads
	for pool_layer_patch in layer.pool_layer_patches
		#THIS IS PROBABLY WRONG SINCE IT DESTROYS TIME SCALE
		#if norm(pool_layer_patch.a_pre) != 0.
	 	update_layer_parameters!(pool_layer_patch)
		#end
	end
end



# PAY ATTENTION: NONLINEARITY SHOULD BE LINEAR IN THIS CASE!!!
@inline function update_layer_parameters!(layer::layer_pool; lc_forward = true) #false : reproduced Földiaks bars
	lc_forward ? update_layer_parameters_lc!(layer) : layer.parameters.updaterule(layer)
end

@inline function lateral_competition!(w, a, lr)
    n, m = size(w)
    dw = similar(w)
    tmp = zeros(m)
    for i in 1:n
        for j in 1:m
            tmp[j] += a[i] * w[i, j]
            dw[i, j] = a[i] * tmp[j]
        end
    end
    BLAS.axpy!(-lr, dw, w)
end
@inline function GH_PCA_Oja!(layer::layer_pool)
	# First: Second term of update rule: "weight-decay" prop. to OLD WEIGHTS
	scale!((1-layer.parameters.learningrate*layer.a.^2),layer.w)
	# Second: First term (data-driven) of weight update
	BLAS.ger!(layer.parameters.learningrate,layer.a,layer.a_pre,layer.w)
end
@inline function GH_PCA_Sanger!(layer::layer_pool)
	# First: Second term of update rule: "weight-decay" prop. to old weights + lateral competition!
	lateral_competition!(layer.w, layer.a, layer.parameters.learningrate)
	# Second: First term (data-driven) of weight update
	BLAS.ger!(layer.parameters.learningrate,layer.a,layer.a_pre,layer.w)
end
@inline function GH_SFA_Oja!(layer::layer_pool)
	scale!((1-layer.parameters.learningrate*layer.a_tr.^2),layer.w)
	BLAS.ger!(layer.parameters.learningrate,layer.a_tr,layer.a_pre,layer.w)
end
@inline function GH_SFA_Sanger!(layer::layer_pool)
	lateral_competition!(layer.w, layer.a_tr, layer.parameters.learningrate)
	BLAS.ger!(layer.parameters.learningrate,layer.a_tr,layer.a_pre,layer.w)
end
# TODO implement subtraction of pre-synaptic trace to avoid permanently active neurons!
@inline function GH_SFA_subtractrace_Oja!(layer::layer_pool)
	scale!((1-layer.parameters.learningrate*layer.a_tr.^2),layer.w)
	BLAS.ger!(layer.parameters.learningrate,layer.a_tr,layer.a_pre-layer.a_tr_pre,layer.w)
end
@inline function GH_SFA_subtractrace_Sanger!(layer::layer_pool)
	lateral_competition!(layer.w, layer.a_tr, layer.parameters.learningrate)
	BLAS.ger!(layer.parameters.learningrate,layer.a_tr,layer.a_pre-layer.a_tr_pre,layer.w)
end
