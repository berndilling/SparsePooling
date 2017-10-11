
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
	layer_post.v += lr_v*(layer_post.a*layer_post.a'-layer_post.p^2)
	for j in 1:size(layer_post.v)[1]
		layer_post.v[j,j] = 0. #no self-inhibition
	end
	clamp!(layer_post.v,0.,Inf64) #Dale's law (Is satisfied automatically as it seems, just to be sure!)

	#Update input weight matrix
	#layer_post.w += lr_w*layer_post.a*(layer_pre.a-layer_post.a*W) # with weight decay... or explicit weight normalization/homeostasis
	BLAS.ger!(lr_w,layer_post.a,layer_pre.a,layer_post.w) #layer_post.w += lr_w*layer_post.a*(layer_pre.a-layer_post.a*W)
	_normalize_inputweights!(layer_post.w) # explicit weight normalization/homeostasis

	#Update thresholds
	layer_post.t += lr_thr*(layer_post.a-layer_post.p)
end

#BRITOS algorithm?

#Algorithm for parameter update for pooling layers with PCA (Oja's rule)
function update_layer_parameters_pool_PCA!(layer_pre, layer_post::layer_pool, learningrate)

end

#Algorithm for parameter update for pooling layers with trace rule/Slow feature analysis
function update_layer_parameters_pool_SLA!(layer_pre, layer_post::layer_pool, learningrate)

end
