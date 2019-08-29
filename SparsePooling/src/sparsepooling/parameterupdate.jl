
#####################################################
#Weight/Parameter update functions

update_layer_parameters!(layer::classifier, data) = _update_layer_parameters!(layer, data)
update_layer_parameters!(layer, data) = _update_layer_parameters!(layer)


@inline function _normalize_inputweights!(weights)
	for j in 1:size(weights)[1]
		weights[j,:] *= 1. / norm(weights[j,:])
	end
end

############################################################################
## Classifier
############################################################################


function _update_layer_parameters!(net::classifier, data; learningrate = 1. / length(net.a_pre), # 0.1 / ...
				   nonlinearity_diff = [relu_diff! for i in 1:net.nl],
				   set_error_lossderivative = _seterror_mse!) # or _seterror_crossentropysoftmax!)
   	target = getlabel(data)
   	set_error_lossderivative(net, target)
	for i in net.nl:-1:2
		nonlinearity_diff[i](net.u[i], net.e[i])
		BLAS.gemv!('T', 1., net.w[i], net.e[i], 0., net.e[i-1])
	end
	nonlinearity_diff[1](net.u[1], net.e[1])
	for i in 1:net.nl
		BLAS.ger!(learningrate, net.e[i], net.a[i], net.w[i])
		BLAS.axpy!(learningrate, net.e[i], net.b[i])
	end
end

############################################################################
## Sparse layer
############################################################################

# Update rule for sparse coding algorithm: meant for both, sparse and pooling layers
#Algorithm for parameter update in sparse coding as proposed by Zylberberg et al PLoS Comp Bio 2011
# Parameters in this paper (lr_v,lr_w,lr_thr)=(0.1,0.001,0.01) (started with higher rates and then decreased)
@inline function update_recurrent_weights!(lr, p, post, v)
	BLAS.ger!(lr, post, post, v)
	@. v += - lr * p^2
	for j in 1:size(v)[1]
		v[j,j] = 0. #no self-inhibition
	end
	clamp!(v, 0., Inf64) # Dale's Law
end
@inline function update_ff_weights!(lr, post, pre, w)
	# First: second term of weight update: weight decay with OLD WEIGHTS à la Oja which comes out of learning rule
	temp = Diagonal(1 .- lr * post) * w
	@. w = temp
	# Second: First term (data-driven) of weight update
	BLAS.ger!(lr, post, pre, w)
	# TODO for sparse layer after non-input layer?
	#BLAS.ger!(layer.parameters.learningrate_w,layer.a,layer.a_pre-layer.a_tr_pre,layer.w)
end
@inline function update_thresholds!(lr, p, post, t)
	BLAS.axpy!(lr, post .- p, t)
	#BLAS.axpy!(layer.parameters.learningrate_thr,Float64.(layer.a .> layer.t) .- p, layer.t)
end
@inline function update_layer_parameters_lc!(layer::layer_sparse)
		update_recurrent_weights!(layer.parameters.learningrate_v, layer.parameters.p, layer.a, layer.v)
		update_ff_weights!(layer.parameters.learningrate_w, layer.a, layer.a_pre, layer.w)
		update_thresholds!(layer.parameters.learningrate_thr, layer.parameters.p, layer.a, layer.t)
end
@inline function update_layer_parameters_lc!(layer::layer_pool)
	update_recurrent_weights!(layer.parameters.learningrate_v, layer.parameters.p, layer.a, layer.v)
	#update_recurrent_weights!(layer.parameters.learningrate_v, layer.parameters.p, layer.a_tr, layer.v)
	#update_recurrent_weights!(layer.parameters.learningrate_v, layer.parameters.p, layer.a_tr-layer.a, layer.v)


	update_ff_weights!(layer.parameters.learningrate_w, layer.a_tr, layer.a_pre - layer.a_tr_pre, layer.w)
	#update_ff_weights!(layer.parameters.learningrate_w, layer.a_tr, layer.a_pre, layer.w)

	#update_ff_weights!(layer.parameters.learningrate_w, layer.a_tr- layer.a, layer.a_pre - layer.a_tr_pre, layer.w)
	#update_ff_weights!(layer.parameters.learningrate_w, layer.a_tr,
	#	(layer.a_pre-layer.a_tr_pre) .* (round.(layer.a_pre) + round.(layer.a_tr_s_pre) .!= 2), layer.w)


	update_thresholds!(layer.parameters.learningrate_thr, layer.parameters.p, layer.a, layer.t)
	#update_thresholds!(layer.parameters.learningrate_thr, layer.parameters.p, layer.a_tr, layer.t)
	#update_thresholds!(layer.parameters.learningrate_thr, layer.parameters.p, layer.a - layer.a_tr, layer.t)
end

@inline function _update_layer_parameters!(layer::layer_sparse)
	if norm(layer.a_pre) != 0. #don't do anything if no input is provided (otherwise thresholds are off)
		update_layer_parameters_lc!(layer)
	end
end
# PAY ATTENTION: lc_forward has to be consistent with the one in forwardprop!
@inline function _update_layer_parameters!(layer::layer_pool; lc_forward = true) #with false : reproduced Földiaks bars
	if norm(layer.a_pre) != 0. #don't do anything if no input is provided
		lc_forward ? update_layer_parameters_lc!(layer) : layer.parameters.updaterule(layer)
	end
end
@inline function _update_layer_parameters!(layer::layer_patchy)
	for layer_patch in layer.layer_patches
	 	_update_layer_parameters!(layer_patch)
	end
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
