
#####################################################
#Weight/Parameter update functions

function _normalize_inputweights!(weights)
	for j in 1:size(weights)[1]
		weights[j,:] *= 1./norm(weights[j,:])
	end
end

#Algorithm for sparse coding as proposed by Zylberberg et al PLoS Comp Bio 2011
function update_layer_parameter_sparse!(layer_pre,layer_post::layer_sparse, learningrate)
	#Update lateral inhibition matrix
	layer_post.v += lr_inh*(layer_pre.a*layer_post.a-layer_post.p^2)
	#Update input weight matrix
	layer_post.w += lr_w*layer_post.a*(layer_pre.a-layer_post.a*W) #weight decay... or explicit weight normalization/homeostasis
	#Update thresholds
	layer_post.t += lr_thr*(layer_post.a-layer_post.p)

	#old code for inspiration:
	#BLAS.ger!(learningrate[i],net.x[i+1],(net.x[i]-BLAS.gemv('T',net.w[i],net.x[i+1])),net.w[i])
	#_normalize_inputweights!(net.w[i])
end
