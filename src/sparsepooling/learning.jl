

using ProgressMeter

function assigninput!(layer,inputfunction,i,order)
	if order == "random"
		input = inputfunction()
	elseif order == "ordered"
		input = inputfunction(i)
	end
		if norm(input) == 0
			assigninput!(layer,inputfunction,i,order)
		else
			layer.a = input
		end
end

#learning sparse layer (post)
function learn_layer_sparse!(layer_pre,
				layer_post::layer_sparse,
				inputfunction::Function,
				iterations::Int64;
				order = "random",
				evaluate_loss_boolian = false,#true,
				nr_evaluations = 100)
				learn_layer_SC!(layer_pre,
								layer_post,
								inputfunction,
								iterations,
								order = order,
								evaluate_loss_boolian = evaluate_loss_boolian,
								nr_evaluations = nr_evaluations)
end
function learn_layer_SC!(layer_pre,
				layer_post,
				inputfunction::Function,
				iterations::Int64;
				order = "random",
				evaluate_loss_boolian = false,#true,
				nr_evaluations = 100)

	print("learning sparse layer...\n")
	ff_boolian = false
	squared_errors = zeros(2,nr_evaluations+1) # values of squared reconstruction error
	#feedforward_differences = zeros(length(layer_post.u),iterations) # difference between pure feedworward and recurrent feedforward
	@showprogress for i in 1:iterations
		#CAUTION: This could result in problems when multiple layers are learnt: activities are overwritten! added 6.11.17
		assigninput!(layer_pre,inputfunction,i,order)
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

function learn_layer_sparse_patchy!(layer_pre,
			layer_post::layer_sparse_patchy,
			iterations::Int64)

	@showprogress for k in 1:iterations
		pattern = get_connected_pattern()
		patches = cut_pattern(pattern)
		for i in 1:layer_post.parameters.n_of_sparse_layer_patches
			if norm(patches[:,:,i]) != 0
				layer_pre.a = patches[:,:,i][:]
				layer_post.sparse_layer_patches[i].u = zeros(length(layer_post.sparse_layer_patches[i].u)) # reset membrane potential
				layer_post.sparse_layer_patches[i].a = zeros(length(layer_post.sparse_layer_patches[i].a)) # reset activities
				forwardprop_lc!(layer_pre, layer_post.sparse_layer_patches[i])
				update_layer_parameters_lc!(layer_pre, layer_post.sparse_layer_patches[i])
			end
		end
	end
end

function learn_layer_pool_patchy!(layer_pre,
			layer_post::layer_pool_patchy,
			n_of_moving_patterns::Int64)


	@showprogress for k in 1:n_of_moving_patterns
		pattern = get_connected_pattern()
		moving_pattern = get_moving_pattern(pattern)
		for j in 1:layer_post.parameters.n_of_sparse_layer_patches
			for i in 1:size(moving_pattern)[3]
				patches = cut_pattern(moving_pattern[:,:,i])

# TODO Here the forwardprop!(net) function is needed!

				forwardprop_lc!(network.layers[1], network.layers[2].sparse_layer_patches[j], patches)
				#subtractmean!(network.layers[2].a)
				forwardprop!(network.layers[2].sparse_layer_patches[j], network.layers[3].pool_layer_patches[j])
				update_layer_parameters_pool!(network.layers[2].sparse_layer_patches[j], network.layers[3].pool_layer_patches[j])
			end
		end
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
		#CAUTION: This could result in problems when multiple layers are learnt: activities are overwritten!
		layer_pre.a = inputfunction(i) #CAUTION: IMAGES PRESENTED IN FIXED ORDER!
		forwardprop!(layer_pre, layer_post) #linear (without non-lin nor biases for PCA)
		update_layer_parameters_pool!(layer_pre, layer_post)
		if evaluate_loss_boolian
			evaluate_loss(layer_pre, layer_post, i, iterations, nr_evaluations, squared_errors)
		end
	end
	return squared_errors
end

function learn_layer_pool_patchy!(network,
				n_of_moving_patterns::Int64)

	@showprogress for k in 1:n_of_moving_patterns
		pattern = get_connected_pattern()
		moving_pattern = get_moving_pattern(pattern)
		for i in 1:size(moving_pattern)[3]
			patches = cut_pattern(moving_pattern[:,:,i])
			forwardprop!(network.layers[1], network.layers[2], patches)
			#subtractmean!(network.layers[2].a)
			forwardprop!(network.layers[2], network.layers[3])
			update_layer_parameters_pool!(network.layers[2], network.layers[3])
		end
	end
end
