
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

function generatehiddenreps_sparse(layer_pre, layer_post::layer_sparse)

end
