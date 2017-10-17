

using ProgressMeter

#learning sparse layer (post)
function learn_layer_sparse!(layer_pre,
				layer_post::layer_sparse,
				inputfunction::Function,
				iterations::Int64)

	@showprogress for i in 1:iterations
		layer_pre.a = inputfunction() #CAUTION: This could result in problems when multiple layers are learnt: activities are overwritten!
		forwardprop!(layer_pre, layer_post)
		update_layer_parameters_sparse!(layer_pre, layer_post)
	end
end

#learning pool layer (post)
function learn_layer_pool!(layer_pre,
				layer_post::layer_pool,
				inputfunction::Function,
				iterations::Int64)

	@showprogress for i in 1:iterations
		layer_pre.a = inputfunction() #CAUTION: This could result in problems when multiple layers are learnt: activities are overwritten!
		forwardprop!(layer_pre, layer_post) #linear (without non-lin nor biases for PCA)
		update_layer_parameters_pool_PCA!(layer_pre, layer_post)
	end
end
