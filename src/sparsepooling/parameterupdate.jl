
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
@inline function update_layer_parameters!(layer_pre, layer_post::layer_sparse)
	 update_layer_parameters_lc!(layer_pre, layer_post)
end
@inline function update_layer_parameters_lc!(layer_pre, layer_post::layer_sparse)
	#Update lateral inhibition matrix
	#layer_post.v += lr_v*(layer_post.a*layer_post.a'-layer_post.parameters.p^2)
	BLAS.ger!(layer_post.parameters.learningrate_v,layer_post.a,layer_post.a,layer_post.v)
	layer_post.v += -layer_post.parameters.learningrate_v*layer_post.parameters.p^2
	for j in 1:size(layer_post.v)[1]
		layer_post.v[j,j] = 0. #no self-inhibition
	end
	clamp!(layer_post.v,0.,Inf64) #Dale's law

	#Update input weight matrix
	#Learning rule:
	#layer_post.w += lr_w*layer_post.a*(layer_pre.a-layer_post.a*W) # with weight decay... or explicit weight normalization/homeostasis
	#Optimized:
	# First: second term of weight update: weight decay with OLD WEIGHTS à la Oja which comes out of learning rule
	scale!((1-layer_post.parameters.learningrate_w*layer_post.a.^2),layer_post.w)
	# Second: First term (data-driven) of weight update
	BLAS.ger!(layer_post.parameters.learningrate_w,layer_post.a,layer_pre.a,layer_post.w)

	#_normalize_inputweights!(layer_post.w) # explicit weight normalization/homeostasis

	#Update thresholds
	#layer_post.t += lr_thr*(layer_post.a-layer_post.parameters.p)
	BLAS.axpy!(layer_post.parameters.learningrate_thr,layer_post.a-layer_post.parameters.p,layer_post.t)
	#avoid negative thesholds:
	#clamp!(layer_post.t,0.,Inf64)
end
@inline function update_layer_parameters_lc!(layer_pre, layer_post::layer_pool)
	BLAS.ger!(layer_post.parameters.learningrate_v,layer_post.a_tr,layer_post.a_tr,layer_post.v)
	#BLAS.ger!(layer_post.parameters.learningrate_v,layer_post.a,layer_post.a,layer_post.v)
	layer_post.v += -layer_post.parameters.learningrate_v*layer_post.parameters.p^2
	for j in 1:size(layer_post.v)[1]
		layer_post.v[j,j] = 0. #no self-inhibition
	end
	clamp!(layer_post.v,0.,Inf64) #Dale's law
	scale!((1-layer_post.parameters.learningrate_w*layer_post.a_tr.^2),layer_post.w)
	#scale!((1-layer_post.parameters.learningrate_w*layer_post.a.^2),layer_post.w)
	BLAS.ger!(layer_post.parameters.learningrate_w,layer_post.a_tr,layer_pre.a,layer_post.w)
	#BLAS.ger!(layer_post.parameters.learningrate_w,layer_post.a,layer_pre.a,layer_post.w)
	BLAS.axpy!(layer_post.parameters.learningrate_thr,layer_post.a_tr-layer_post.parameters.p,layer_post.t)
	#BLAS.axpy!(layer_post.parameters.learningrate_thr,layer_post.a-layer_post.parameters.p,layer_post.t)
	#clamp!(layer_post.t,0.,Inf64)
end

@inline function update_layer_parameters!(layer_pre, layer_post::layer_sparse_patchy)
	for sparse_layer_patch in layer_post.sparse_layer_patches
	 	update_layer_parameters_lc!(layer_pre, sparse_layer_patch)
	end
end

# TODO BRITOS algorithm?

############################################################################
## Pooling layer
############################################################################

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

# PAY ATTENTION: NONLINEARITY SHOULD BE LINEAR IN THIS CASE!!!
@inline function update_layer_parameters!(layer_pre, layer_post::layer_pool)
	layer_post.parameters.updaterule(layer_pre, layer_post)
end
@inline function GH_PCA_Oja!(layer_pre, layer_post::layer_pool)
	# First: Second term of update rule: "weight-decay" prop. to OLD WEIGHTS
	scale!((1-layer_post.parameters.learningrate*layer_post.a.^2),layer_post.w)
	# Second: First term (data-driven) of weight update
	BLAS.ger!(layer_post.parameters.learningrate,layer_post.a,layer_pre.a,layer_post.w)
end
@inline function GH_PCA_Sanger!(layer_pre, layer_post::layer_pool)
	# First: Second term of update rule: "weight-decay" prop. to old weights + lateral competition!
	lateral_competition!(layer_post.w, layer_post.a, layer_post.parameters.learningrate)
	# Second: First term (data-driven) of weight update
	BLAS.ger!(layer_post.parameters.learningrate,layer_post.a,layer_pre.a,layer_post.w)
end
@inline function GH_SFA_Oja!(layer_pre, layer_post::layer_pool)
	scale!((1-layer_post.parameters.learningrate*layer_post.a_tr.^2),layer_post.w)
	BLAS.ger!(layer_post.parameters.learningrate,layer_post.a_tr,layer_pre.a,layer_post.w)
end
@inline function GH_SFA_Sanger!(layer_pre, layer_post::layer_pool)
	lateral_competition!(layer_post.w, layer_post.a_tr, layer_post.parameters.learningrate)
	BLAS.ger!(layer_post.parameters.learningrate,layer_post.a_tr,layer_pre.a,layer_post.w)
end
# TODO implement subtraction of pre-synaptic trace to avoid permanently active neurons!
@inline function GH_SFA_subtractrace_Oja!(layer_pre, layer_post::layer_pool)
	scale!((1-layer_post.parameters.learningrate*layer_post.a_tr.^2),layer_post.w)
	BLAS.ger!(layer_post.parameters.learningrate,layer_post.a_tr,layer_pre.a-layer_pre.a_tr,layer_post.w)
end
@inline function GH_SFA_subtractrace_Sanger!(layer_pre, layer_post::layer_pool)
	lateral_competition!(layer_post.w, layer_post.a_tr, layer_post.parameters.learningrate)
	BLAS.ger!(layer_post.parameters.learningrate,layer_post.a_tr,layer_pre.a-layer_pre.a_tr,layer_post.w)
end
