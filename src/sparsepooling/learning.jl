

using ProgressMeter

#learning sparse layer (post)
function learn_layer_sparse!(layer_pre,
				layer_post::layer_sparse,
				inputfunction::Function,
				iterations::Int64;
				evaluate_loss_boolian = false,#true,
				nr_evaluations = 100)
				learn_layer_SC!(layer_pre,
								layer_post,
								inputfunction,
								iterations,
								evaluate_loss_boolian = evaluate_loss_boolian,
								nr_evaluations = nr_evaluations)
end
function learn_layer_SC!(layer_pre,
				layer_post,
				inputfunction::Function,
				iterations::Int64;
				evaluate_loss_boolian = false,#true,
				nr_evaluations = 100)

	print("learning sparse layer...\n")
	ff_boolian = true
	squared_errors = zeros(2,nr_evaluations+1) # values of squared reconstruction error
	feedforward_differences = zeros(length(layer_post.u),iterations) # difference between pure feedworward and recurrent feedforward
	@showprogress for i in 1:iterations
		layer_pre.a = inputfunction()#inputfunction(i) #CAUTION: This could result in problems when multiple layers are learnt: activities are overwritten!
		#added 6.11.17
		layer_post.u = zeros(length(layer_post.u)) # reset membrane potential
		layer_post.a = zeros(length(layer_post.a)) # reset activities
		forwardprop_lc!(layer_pre, layer_post)
		#feedforward_differences[:,i] = evaluate_ff_difference(layer_pre, layer_post)
		update_layer_parameters_lc!(layer_pre, layer_post)
		if evaluate_loss_boolian #ATTENTION: NOT REAL LOSS FUNCTION FOR SPARSE CODING! ONLY RECONSTRUCTION ERROR!
			evaluate_loss(layer_pre, layer_post, i, iterations, nr_evaluations, squared_errors)
		end
	end
	if ff_boolian
		return squared_errors, feedforward_differences
	else
		return squared_errors
	end
end

#learning pool layer (post)
function learn_layer_pool!(layer_pre,
				layer_post::layer_pool,
				inputfunction::Function,
				iterations::Int64;
				evaluate_loss_boolian = false,
				nr_evaluations = 20)

	print("learning pooling layer...\n")
	squared_errors = zeros(2,nr_evaluations+1) # values of squared reconstruction error
	@showprogress for i in 1:iterations
		layer_pre.a = inputfunction(i) #CAUTION: This could result in problems when multiple layers are learnt: activities are overwritten!
		forwardprop!(layer_pre, layer_post) #linear (without non-lin nor biases for PCA)
		update_layer_parameters_pool!(layer_pre, layer_post)
		if evaluate_loss_boolian
			evaluate_loss(layer_pre, layer_post, i, iterations, nr_evaluations, squared_errors)
		end
	end
	return squared_errors
end
