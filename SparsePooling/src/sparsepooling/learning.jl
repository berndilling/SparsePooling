
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

function learn_net_layerwise!(net::net,
	data,
	intermediatestates,
	iterations::Array{Int64, 1},
	inputfunctions,
	dynamicfunctions;
	LearningFromLayer = 2, LearningUntilLayer = net.nr_layers, cut_size = 0, eval_loss = false)
	# cut_size only used for single patch learning, see e.g. getmovingimagepatch()

	print(string("\n Learn network layers ",LearningFromLayer, " to ",LearningUntilLayer,"\n"))
	losses = []
	for k in LearningFromLayer:LearningUntilLayer
		loss = []
		print(string("\n Learning Layer Nr. ",k," (",typeof(net.layers[k]),")\n"))
		@showprogress for i in 1:iterations[k - 1]
			pattern = inputfunctions[k-1](data)
			dynamicpattern = dynamicfunctions[k-1](pattern; cut_size = cut_size)
			for j in 1:size(dynamicpattern)[3]
				net.layers[1].a = dynamicpattern[:,:,j][:]
				forwardprop!(net; FPUntilLayer = k)
				update_layer_parameters!(net.layers[k], data)
			end
			eval_loss && push!(loss, get_loss(net.layers[k]))
		end
		eval_loss && push!(losses, loss)
		push!(intermediatestates,deepcopy(net))
	end
	return losses
end
export learn_net_layerwise!
