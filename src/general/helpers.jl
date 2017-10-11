
#####################################################
#Helpers

function getsparsity(input::Array{Float64, 1})
	length(find(x -> (x == 0),input))/length(input)
end

function getsmallimg()
    global patternindex = rand(1:size(smallimgs)[2])
    smallimgs[:, patternindex]
end

function getsample()
    global patternindex = rand(1:n_samples)
    smallimgs[:, patternindex]
end

function getlabel(x)
    [labels[patternindex] == i for i in 0:9]
end

function gethiddenreps()
		global patternindex = rand(1:size(smallimgs)[2])
		#sae.reps[n_ae][1][:, patternindex]
		sae.reps[n_ae-1][1][:, patternindex]
end

function gethiddenrepstest()
		global patternindex = rand(1:size(smallimgstest)[2])
		#sae.reps[n_ae][1][:, patternindex]
		sae.reps[n_ae-1][2][:, patternindex]
end

function generatehiddenreps_sparse(layer_pre, layer_post::layer_sparse, data, storage)
	for i in 1:size(data)[2]
		layer_pre.a = data[:,i]
		forwardprop!(layer_pre, layer_post)
		storage[i,:] = layer_post.a
	end
end
