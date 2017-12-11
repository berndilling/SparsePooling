
using Distributions
using ProgressMeter

#####################################################
#Helpers

function getsparsity(input::Array{Float64, 1})
	length(find(x -> (x == 0),input))/length(input)
end

function getsmallimg()
    patternindex = rand(1:size(smallimgs)[2])
    smallimgs[:, patternindex]
end

function getsmallimg(iteration) # to avoid conflicts when switching from moving bars
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

#get bars moving in one direction 1 pixel per timestep/iteration
#smallimgs should be array of bars with (used) length "length"
#repetitions: same bar will be presented for ("repetitions") subsequent frames
function get_moving_bar(iteration; repetitions = 1)
		smallimgs[:,Int(ceil((iteration/repetitions-1) % 24)) + 1]
end
function get_moving_vbar(iteration; repetitions = 1)
		smallimgs[:,Int(ceil((iteration/repetitions-1) % 12)) + 1]
end
function get_moving_hbar(iteration; repetitions = 1)
		smallimgs[:,Int(ceil((iteration/repetitions-1) % 12)) + 12]
end
function get_jittered_bar(iteration; repetitions_per_orientation = 30)
		if Int(ceil(iteration/repetitions_per_orientation)-1) % 2 == 0
			smallimgs[:,rand(1:12)]
		else#if iteration % repetitions_per_orientation >= 0
			smallimgs[:,rand(13:24)]
		end
end

function getsavepath()
	if is_apple()
		path = "/Users/Bernd/Documents/PhD/Projects/"
	elseif is_linux()
		path = "/home/illing/"
	end
end

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

#up to now: only squared reconstruction error!
#for SC: full loss function:
#losses[i] = squared_errors[i] + sum(layer_post.a)-length(layer_post.a)*p + sum(layer_post.a*layer_post.a')-length(layer_post.a)*p^2
function evaluate_loss(layer_pre, layer_post, i, iterations, nr_evaluations, squared_errors)
	if i == 1
		squared_errors[:,1] = _evaluate_errors(layer_pre,layer_post,i)
	elseif i % Int(iterations/nr_evaluations) == 0
		squared_errors[:,Int(i*nr_evaluations/iterations)+1] = _evaluate_errors(layer_pre,layer_post,i)
	end
end

# to evaluate difference between pure feedforward and recurrent sparse coding feed-forward
function evaluate_ff_difference(layer_pre, layer_post::layer_sparse)
	layer_post.u-BLAS.gemv('N',layer_post.w,layer_pre.a)
end

function microsaccade(imagevector; max_amplitude = 3)
	dim = Int(sqrt(length(imagevector)))
	amps = rand(-max_amplitude:max_amplitude,2) #draw random translation
	circshift(reshape(imagevector,dim,dim),amps)[:]
end

# generate superimposed bars
# data: array with vertical and horizontal bars (144*24-array)
function create_superimposed_bars(bars; prob = 1/12, nr_examples = 50000)
	data = zeros(144,nr_examples)
	for i in 1:nr_examples
		select = rand(Bernoulli(prob),24)
		data[:,i] = clamp(sum(bars*Diagonal(select),2),0,1)
	end
	return data
end

function set_init_bars!(layer::layer_sparse,hidden_size)
	layer.w = rand(size(layer.w)[1],size(layer.w)[2])/hidden_size
end
