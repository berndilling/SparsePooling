

using ProgressMeter

# function assigninput!(layer,inputfunction,i,order)
# 	if order == "random"
# 		input = inputfunction()
# 	elseif order == "ordered"
# 		input = inputfunction(i)
# 	end
# 		if norm(input) == 0
# 			assigninput!(layer,inputfunction,i,order)
# 		else
# 			layer.a = input
# 		end
# end
#
# #learning sparse layer (post)
# function learn_layer_sparse!(layer_pre,
# 				layer_post::layer_sparse,
# 				inputfunction::Function,
# 				iterations::Int64;
# 				order = "random",
# 				evaluate_loss_boolian = false,#true,
# 				nr_evaluations = 100)
# 				learn_layer_SC!(layer_pre,
# 								layer_post,
# 								inputfunction,
# 								iterations,
# 								order = order,
# 								evaluate_loss_boolian = evaluate_loss_boolian,
# 								nr_evaluations = nr_evaluations)
# end
# function learn_layer_SC!(layer_pre,
# 				layer_post,
# 				inputfunction::Function,
# 				iterations::Int64;
# 				order = "random",
# 				evaluate_loss_boolian = false,#true,
# 				nr_evaluations = 100)
#
# 	print("learning sparse layer...\n")
# 	ff_boolian = false
# 	squared_errors = zeros(2,nr_evaluations+1) # values of squared reconstruction error
# 	#feedforward_differences = zeros(length(layer_post.u),iterations) # difference between pure feedworward and recurrent feedforward
# 	@showprogress for i in 1:iterations
# 		#CAUTION: This could result in problems when multiple layers are learnt: activities are overwritten! added 6.11.17
# 		assigninput!(layer_pre,inputfunction,i,order)
# 		layer_post.u = zeros(length(layer_post.u)) # reset membrane potential
# 		layer_post.a = zeros(length(layer_post.a)) # reset activities
# 		forwardprop_lc!(layer_pre, layer_post)
# 		#feedforward_differences[:,i] = evaluate_ff_difference(layer_pre, layer_post)
# 		update_layer_parameters_lc!(layer_pre, layer_post)
# 		if evaluate_loss_boolian #ATTENTION: NOT REAL LOSS FUNCTION FOR SPARSE CODING! ONLY RECONSTRUCTION ERROR!
# 			evaluate_loss(layer_pre, layer_post, i, iterations, nr_evaluations, squared_errors)
# 		end
# 	end
# 	if ff_boolian
# 		return squared_errors, feedforward_differences
# 	else
# 		return squared_errors
# 	end
# end

###################################################################################

function learn_layer_sparse_patchy!(layer_pre::layer_input,
			layer_post::layer_sparse_patchy,
			iterations::Int64;
			inputfunction = get_connected_pattern,
			dynamicfunction = staticpattern)

	@showprogress for k in 1:iterations
		pattern = inputfunction()
		dynamicpattern = dynamicfunction(pattern)
		for j in 1:size(dynamicpattern)[3]
			patches = cut_pattern(dynamicpattern[:,:,j])
			i = 1
			# TODO parallelize this! see Eventbasedlifintergator! Common layer_pre is a problem!
			for sparse_layer_patch in layer_post.sparse_layer_patches
				if norm(patches[:,:,i]) != 0
					layer_pre.a = patches[:,:,i][:]
					sparse_layer_patch.u = zeros(length(sparse_layer_patch.u)) # reset membrane potential
					sparse_layer_patch.a = zeros(length(sparse_layer_patch.a)) # reset activities
					forwardprop!(layer_pre, sparse_layer_patch)
					update_layer_parameters!(layer_pre, sparse_layer_patch)
				end
				i += 1
			end
		end
	end
end

function learn_layer_pool!(layer_pre,
			layer_post::layer_pool,
			n_of_moving_patterns::Int64)

	@showprogress for k in 1:n_of_moving_patterns
		pattern = get_connected_pattern()
		moving_pattern = get_moving_pattern(pattern)
		for i in 1:size(moving_pattern)[3]
			patches = cut_pattern(moving_pattern[:,:,i])
			forwardprop!(network.layers[1], network.layers[2], patches)
			forwardprop!(network.layers[2], network.layers[3])
			update_layer_parameters_pool!(network.layers[2], network.layers[3])
		end
	end
end

################################################################################
# Network level learning (layer-wise)


function learn_net_layerwise!(net::net,intermediatestates,
	iterations::Array{Int64, 1},
	inputfunctions,
	dynamicfunctions;
	LearningFromLayer = 2, LearningUntilLayer = net.nr_layers)
	print(string("\n Learn network layers ",LearningFromLayer, " to ",LearningUntilLayer,"\n"))
	for k in LearningFromLayer:LearningUntilLayer
		print(string("\n Learning Layer Nr. ",k," (",typeof(net.layers[k]),")\n"))
		if typeof(net.layers[k]) == layer_sparse_patchy
			 if k == 2
				 learn_layer_sparse_patchy!(net.layers[k - 1],net.layers[k],
			 			iterations[k - 1];
			 			inputfunction = inputfunctions[k - 1],
			 			dynamicfunction = dynamicfunctions[k - 1])
			else
				error("Framework not ready for patchy_SC in non-input layer!!!")
			end
		else
			@showprogress for i in 1:iterations[k - 1]
				pattern = inputfunctions[k-1]()
				dynamicpattern = dynamicfunctions[k-1](pattern)
				for j in 1:size(dynamicpattern)[3]
					patches = cut_pattern(dynamicpattern[:,:,j])
					forwardprop!(net, patches; FPUntilLayer = k)
					update_layer_parameters!(net.layers[k - 1], net.layers[k])
				end
			end
			push!(intermediatestates,deepcopy(net))
		end
	end
end
