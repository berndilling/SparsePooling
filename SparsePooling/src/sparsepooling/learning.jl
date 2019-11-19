
using ProgressMeter

@inline function get_reconstruction_loss(x, y, w)
	return copy(norm(x .- w' * y))
end
@inline function get_loss(layer::layer)
	get_reconstruction_loss(layer.a_pre, layer.a, layer.w)
end
@inline function get_loss(layer::layer_patchy)
	get_reconstruction_loss(layer.layer_patches[1])
end

@inline function network_iteration!(net, input, k, data)
	net.layers[1].a = input
	forwardprop!(net; FPUntilLayer = k)
	update_layer_parameters!(net.layers[k], data)
end

function learn_net_layerwise!(net::net,
	data,
	intermediatestates,
	iterations::Array{Int64, 1},
	inputfunctions,
	dynamicfunctions;
	LearningFromLayer = 2, LearningUntilLayer = net.nr_layers, cut_size = 0, eval_loss = false, dynamic = true)
	
	data.color && dynamic && error("Dynamic not implemented for color images!!! Change in learn_net_layerwise!()")

	print(string("\n Learn network layers ",LearningFromLayer, " to ",LearningUntilLayer,"\n"))
	losses = []
	for k in LearningFromLayer:LearningUntilLayer
		loss = []
		print(string("\n Learning Layer Nr. ",k," (",typeof(net.layers[k]),")\n"))
		@showprogress for i in 1:iterations[k - 1]
			pattern = inputfunctions[k-1](data)
			if dynamic # Careful! not implemented for color yet!!!
				dynamicpattern = dynamicfunctions[k-1](data, pattern; cut_size = cut_size)
				for j in 1:size(dynamicpattern)[3]
					network_iteration!(net, dynamicpattern[:,:,j][:], k, data)
				end
			else
				network_iteration!(net, pattern, k, data)
			end
			eval_loss && push!(loss, get_loss(net.layers[k]))
		end
		eval_loss && push!(losses, loss)
		push!(intermediatestates,deepcopy(net))
	end
	return losses
end
export learn_net_layerwise!
