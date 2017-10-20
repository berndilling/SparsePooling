
#####################################################
#Helpers

function getsparsity(input::Array{Float64, 1})
	length(find(x -> (x == 0),input))/length(input)
end

function getsmallimg()
    patternindex = rand(1:size(smallimgs)[2])
    smallimgs[:, patternindex]
end

function getsample()
    patternindex = rand(1:n_samples)
    smallimgs[:, patternindex]
end

function getlabel(x)
    [labels[patternindex] == i for i in 0:9]
end

function getsavepath()
	if is_apple()
		path = "/Users/Bernd/Documents/PhD/Projects/"
	elseif is_linux()
		path = "/home/illing/"
	end
end

using ProgressMeter

function generatehiddenreps(layer_pre, layer_post; number_of_reps = Int(5e4))
	print("\n")
	print(string("Generate ",number_of_reps," hidden representations for layer type: ",typeof(layer_post)))
	print("\n")
	layer_post.hidden_reps = zeros(length(layer_post.a),number_of_reps)
	@showprogress for i in 1:number_of_reps
		layer_pre.a = smallimgs[:,i]
		forwardprop!(layer_pre, layer_post)
		layer_post.hidden_reps[:,i] = deepcopy(layer_post.a)
	end
end

function _evaluate_errors(layer_pre, layer_post, i)
	generatehiddenreps(layer_pre, layer_post)
	return [i,mean((smallimgs[:,1:Int(5e4)] - BLAS.gemm('T', 'N', layer_post.w, layer_post.hidden_reps)).^2)]
end

function evaluate_loss(layer_pre, layer_post, i, iterations, nr_evaluations, squared_errors)
	if i == 1
		squared_errors[:,1] = _evaluate_errors(layer_pre,layer_post,i)
	elseif i % Int(iterations/nr_evaluations) == 0
		squared_errors[:,Int(i*nr_evaluations/iterations)+1] = _evaluate_errors(layer_pre,layer_post,i)
	end
end
