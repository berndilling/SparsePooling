
#####################################################
#Weight/Parameter update functions

function _normalize_inputweights!(weights)
	for j in 1:size(weights)[1]
		weights[j,:] *= 1./norm(weights[j,:])
	end
end

#Algorithm for parameter update in sparse coding as proposed by Zylberberg et al PLoS Comp Bio 2011
function update_layer_parameters_sparse!(layer_pre, layer_post::layer_sparse, learningrate)
	#Update lateral inhibition matrix
	layer_post.v += lr_inh*(layer_pre.a*layer_post.a'-layer_post.p^2)
	for j in 1:size(layer_post.v)[1]
		layer_post.v[j,j] = 0. #no self-inhibition
	end
	#Update input weight matrix
	#layer_post.w += lr_w*layer_post.a*(layer_pre.a-layer_post.a*W) # with weight decay... or explicit weight normalization/homeostasis
	BLAS.ger!(lr_w,layer_post.a,layer_pre.a,layer_post.w) #layer_post.w += lr_w*layer_post.a*(layer_pre.a-layer_post.a*W)
	_normalize_inputweights!(layer_post.w) # explicit weight normalization/homeostasis
	#Update thresholds
	layer_post.t += lr_thr*(layer_post.a-layer_post.p)

	#old code for inspiration:
	BLAS.ger!(learningrate[i],net.x[i+1],(net.x[i]-BLAS.gemv('T',net.w[i],net.x[i+1])),net.w[i])
	#_normalize_inputweights!(net.w[i])
end

#BRITOS algorithm?

#Algorithm for parameter update for pooling layers with PCA (Oja's rule)
function update_layer_parameters_pool_PCA!(layer_pre, layer_post::layer_pool, learningrate)

end

#Algorithm for parameter update for pooling layers with trace rule/Slow feature analysis
function update_layer_parameters_pool_SLA!(layer_pre, layer_post::layer_pool, learningrate)

end
