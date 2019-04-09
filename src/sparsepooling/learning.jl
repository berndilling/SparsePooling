
using ProgressMeter

function learn_net_layerwise!(net::net,intermediatestates,
	iterations::Array{Int64, 1},
	inputfunctions,
	dynamicfunctions;
	LearningFromLayer = 2, LearningUntilLayer = net.nr_layers)
	print(string("\n Learn network layers ",LearningFromLayer, " to ",LearningUntilLayer,"\n"))
	for k in LearningFromLayer:LearningUntilLayer
		print(string("\n Learning Layer Nr. ",k," (",typeof(net.layers[k]),")\n"))
		@showprogress for i in 1:iterations[k - 1]
			pattern = inputfunctions[k-1]()
			dynamicpattern = dynamicfunctions[k-1](pattern)
			for j in 1:size(dynamicpattern)[3]
				net.layers[1].a = dynamicpattern[:,:,j][:]
				forwardprop!(net; FPUntilLayer = k)
				update_layer_parameters!(net.layers[k])
			end
		end
		push!(intermediatestates,deepcopy(net))
	end
end
