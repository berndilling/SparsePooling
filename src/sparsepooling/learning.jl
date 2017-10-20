

using ProgressMeter

#learning sparse layer (post)
function learn_layer_sparse!(layer_pre,
				layer_post::layer_sparse,
				inputfunction::Function,
				iterations::Int64;
				p = 0.05,
				evaluate_loss_boolian = true,
				nr_evaluations = 100)

	squared_errors = zeros(2,nr_evaluations+1) # values of squared reconstruction error
	losses = zeros(2,nr_evaluations+1) # values of loss function (Zylberberg)
	@showprogress for i in 1:iterations
		layer_pre.a = inputfunction() #CAUTION: This could result in problems when multiple layers are learnt: activities are overwritten!
		forwardprop!(layer_pre, layer_post)
		update_layer_parameters_sparse!(layer_pre, layer_post)
		if evaluate_loss_boolian #ATTENTION: NOT REAL LOSS FUNCTION FOR SPARSE CODING! ONLY RECONSTRUCTION ERROR!
			evaluate_loss(layer_pre, layer_post, i, iterations, nr_evaluations, squared_errors)
		end
		#losses[i] = squared_errors[i] + sum(layer_post.a)-length(layer_post.a)*p + sum(layer_post.a*layer_post.a')-length(layer_post.a)*p^2
	end
	return squared_errors#, losses
end

#learning pool layer (post)
function learn_layer_pool!(layer_pre,
				layer_post::layer_pool,
				inputfunction::Function,
				iterations::Int64;
				evaluate_loss_boolian = true,
				nr_evaluations = 20)

	squared_errors = zeros(2,nr_evaluations+1) # values of squared reconstruction error
	@showprogress for i in 1:iterations
		layer_pre.a = inputfunction() #CAUTION: This could result in problems when multiple layers are learnt: activities are overwritten!
		forwardprop!(layer_pre, layer_post) #linear (without non-lin nor biases for PCA)
		update_layer_parameters_pool_PCA!(layer_pre, layer_post)
		if evaluate_loss_boolian
			evaluate_loss(layer_pre, layer_post, i, iterations, nr_evaluations, squared_errors)
		end
	end
	return squared_errors
end
