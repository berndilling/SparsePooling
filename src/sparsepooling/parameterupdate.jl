
#####################################################
#Weight/Parameter update functions

function _normalize_inputweights!(weights)
	for j in 1:size(weights)[1]
		weights[j,:] *= 1./norm(weights[j,:])
	end
end

#Algorithm for parameter update in sparse coding as proposed by Zylberberg et al PLoS Comp Bio 2011
# Parameters in this paper (lr_v,lr_w,lr_thr)=(0.1,0.001,0.01) (started with higher rates and then decreased)
function update_layer_parameters_sparse!(layer_pre, layer_post::layer_sparse; lr_v = 1e-1, lr_w = 1e-3, lr_thr = 1e-2)
	#Update lateral inhibition matrix
	#layer_post.v += lr_v*(layer_post.a*layer_post.a'-layer_post.p^2)
	BLAS.ger!(lr_v,layer_post.a,layer_post.a,layer_post.v)
	layer_post.v += -lr_v*layer_post.p^2
	for j in 1:size(layer_post.v)[1]
		layer_post.v[j,j] = 0. #no self-inhibition
	end
	clamp!(layer_post.v,0.,Inf64) #Dale's law

	#Update input weight matrix
	#Learning rule:
	#layer_post.w += lr_w*layer_post.a*(layer_pre.a-layer_post.a*W) # with weight decay... or explicit weight normalization/homeostasis
	#Optimized:
	# First: second term of weight update: weight decay with OLD WEIGHTS à la Oja which comes out of learning rule
	scale!((1-lr_w*layer_post.a.^2),layer_post.w)
	# Second: First term (data-driven) of weight update
	BLAS.ger!(lr_w,layer_post.a,layer_pre.a,layer_post.w)

	#_normalize_inputweights!(layer_post.w) # explicit weight normalization/homeostasis

	#Update thresholds
	#layer_post.t += lr_thr*(layer_post.a-layer_post.p)
	BLAS.axpy!(lr_thr,layer_post.a-layer_post.p,layer_post.t)
	#avoid negative thesholds:
	#clamp!(layer_post.t,0.,Inf64)
end

#BRITOS algorithm?

#Algorithm for parameter update for pooling layers with PCA (Oja's rule) OR SANGERS RULE!
#PAY ATTENTION: NONLINEARITY SHOULD BE LINEAR IN THIS CASE!!!
function update_layer_parameters_pool_PCA!(layer_pre, layer_post::layer_pool; learningrate = 1e-2, rule = "Sanger")
	if rule == "Oja"
		#Oja's rule (should give 1. principal component for all hidden units)
		# First: Second term of update rule: "weight-decay" prop. to OLD WEIGHTS
		scale!((1-learningrate*layer_post.a.^2),layer_post.w)
		# Second: First term (data-driven) of weight update
		BLAS.ger!(learningrate,layer_post.a,layer_pre.a,layer_post.w)
	elseif rule == "Sanger"
		#Sanger's rule
		# First: Second term of update rule: "weight-decay" prop. to old weights
		# + lateral competition!
		layer_post.w += -learningrate*LowerTriangular(layer_post.a*layer_post.a')*layer_post.w
		# Second: First term (data-driven) of weight update
		BLAS.ger!(learningrate,layer_post.a,layer_pre.a,layer_post.w)
		#BLAS.syr!('L', learningrate, layer_post.a, layer_post.w)
	end
end

#Algorithm for parameter update for pooling layers with trace rule/Slow feature analysis
function update_layer_parameters_pool_SFA!(layer_pre, layer_post::layer_pool; learningrate = 1e-2, rule = "Sanger-like")
	if rule == "Oja-like"
		# First: Second term of update rule: "weight-decay" prop. to OLD WEIGHTS
		scale!((1-learningrate*layer_post.a_tr.^2),layer_post.w)
		# Second: First term (data-driven) of weight update
		BLAS.ger!(learningrate,layer_post.a_tr,layer_pre.a,layer_post.w)
	elseif rule == "Sanger-like"
		#Sanger's rule
		# First: Second term of update rule: "weight-decay" prop. to old weights
		# + lateral competition!
		layer_post.w += -learningrate*LowerTriangular(layer_post.a_tr*layer_post.a_tr')*layer_post.w
		# Second: First term (data-driven) of weight update
		BLAS.ger!(learningrate,layer_post.a_tr,layer_pre.a,layer_post.w)
		#BLAS.syr!('L', learningrate, layer_post.a, layer_post.w)
	end
end
