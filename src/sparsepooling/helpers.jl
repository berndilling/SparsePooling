
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

function _generatehiddenreps!(sae,n_ae::Int64, smallimgs, smallimgstest, conf, func_forwardprop::Function)
	if typeof(conf) == configAE
		if n_ae == 1
			for i in 1:size(smallimgs)[2]
				sae.aes[n_ae].x[1] = smallimgs[:, i]
				func_forwardprop(sae.aes[n_ae], nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][1][:,i] = deepcopy(sae.aes[n_ae].x[Int(mean([1,sae.aes[n_ae].nl+1]))])
			end
			for i in 1:size(smallimgstest)[2]
				sae.aes[n_ae].x[1] = smallimgstest[:, i]
				func_forwardprop(sae.aes[n_ae], nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][2][:,i] = deepcopy(sae.aes[n_ae].x[Int(mean([1,sae.aes[n_ae].nl+1]))])
			end
		else
			for i in 1:size(smallimgs)[2]
				sae.aes[n_ae].x[1] = deepcopy(sae.reps[n_ae-1][1][:,i])
				func_forwardprop(sae.aes[n_ae], nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][1][:,i] = deepcopy(sae.aes[n_ae].x[Int(mean([1,sae.aes[n_ae].nl+1]))])
			end
			for i in 1:size(smallimgstest)[2]
				sae.aes[n_ae].x[1] = deepcopy(sae.reps[n_ae-1][2][:,i])
				func_forwardprop(sae.aes[n_ae], nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][2][:,i] = deepcopy(sae.aes[n_ae].x[Int(mean([1,sae.aes[n_ae].nl+1]))])
			end
		end
	elseif typeof(conf) == configAE_sparse
		if n_ae == 1
			for i in 1:size(smallimgs)[2]
				sae.aes[n_ae].x[1] = smallimgs[:, i]
				func_forwardprop(sae.aes[n_ae], conf.lambda, nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][1][:,i] = deepcopy(sae.aes[n_ae].x[end])
			end
			for i in 1:size(smallimgstest)[2]
				sae.aes[n_ae].x[1] = smallimgstest[:, i]
				func_forwardprop(sae.aes[n_ae], conf.lambda, nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][2][:,i] = deepcopy(sae.aes[n_ae].x[end])
			end
		else
			for i in 1:size(smallimgs)[2]
				sae.aes[n_ae].x[1] = deepcopy(sae.reps[n_ae-1][1][:,i])
				func_forwardprop(sae.aes[n_ae], conf.lambda, nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][1][:,i] = deepcopy(sae.aes[n_ae].x[end])
			end
			for i in 1:size(smallimgstest)[2]
				sae.aes[n_ae].x[1] = deepcopy(sae.reps[n_ae-1][2][:,i])
				func_forwardprop(sae.aes[n_ae], conf.lambda, nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][2][:,i] = deepcopy(sae.aes[n_ae].x[end])
			end
		end
	end
end

function generatehiddenreps!(sae::SAE,n_ae::Int64, smallimgs, smallimgstest, conf::configAE)
	@printf "compute hidden representations (for train and test sets) of AE nr. %.2i ...\n" n_ae
	_generatehiddenreps!(sae,n_ae, smallimgs, smallimgstest, conf, forwardprop!)
end

#for sparse coding (only 2-layer AEs!)
function generatehiddenreps!(sae::SAE_sparse,n_ae::Int64, smallimgs, smallimgstest, conf::configAE_sparse)
	@printf "compute sparse hidden representations (for train and test sets) of AE nr. %.2i ...\n" n_ae
	_generatehiddenreps!(sae,n_ae, smallimgs, smallimgstest, conf, forwardprop!)
end

#for sparse coding à la Brito (only 2-layer AEs!)
function generatehiddenreps_Brito!(sae::SAE_sparse,n_ae::Int64, smallimgs, smallimgstest, conf::configAE_sparse)
	@printf "compute sparse (Britos algorithm) hidden representations (for train and test sets) of AE nr. %.2i ...\n" n_ae
	_generatehiddenreps!(sae,n_ae, smallimgs, smallimgstest, conf, forwardprop_Brito_full!) #forwardprop_Brito!
end

#for unlabelled data
function generatehiddenreps!(sae::SAE,n_ae::Int64, smallimgs, conf::configAE)
	@printf "compute hidden representations (for train and test sets) of AE nr. %.2i ...\n" n_ae
	if n_ae == 1
		for i in 1:size(smallimgs)[2]
			sae.aes[n_ae].x[1] = smallimgs[:, i]
			forwardprop!(sae.aes[n_ae], nonlinearity = conf.nonlinearity)
			sae.reps[n_ae][1][:,i] = deepcopy(sae.aes[n_ae].x[Int(mean([1,sae.aes[n_ae].nl+1]))])
		end
	else
		for i in 1:size(smallimgs)[2]
			sae.aes[n_ae].x[1] = deepcopy(sae.reps[n_ae-1][1][:,i])
			forwardprop!(sae.aes[n_ae], nonlinearity = conf.nonlinearity)
			sae.reps[n_ae][1][:,i] = deepcopy(sae.aes[n_ae].x[Int(mean([1,sae.aes[n_ae].nl+1]))])
		end
	end
end

function geterrors(net, imgs, labels; nonlinearity = nonlinearity)
	error = 0
	noftest = size(imgs)[2]
	for i in 1:noftest
		net.x[1] = imgs[:, i]
		forwardprop!(net, nonlinearity = nonlinearity)
		error += indmax(net.x[end]) != labels[i] + 1
	end
	error/noftest
end

function geterrorsAE(net, imgs; nonlinearity = nonlinearity)
	error = 0
	noftest = size(imgs)[2]
	for i in 1:noftest
		net.x[1] = imgs[:, i]
		forwardprop!(net, nonlinearity = nonlinearity)
		net.e[end] = net.x[1] - net.x[end]
		error += norm(net.e[end])^2
	end
	sqrt(error)/noftest
end


function printerrors(net, smallimgs, labels, smallimgstest, labelstest, conf)
	error_train = geterrors(net, smallimgs, labels, nonlinearity = conf.nonlinearity)
	error_test = geterrors(net, smallimgstest, labelstest, nonlinearity = conf.nonlinearity)
	@printf "%.2f %% on training set\n" 100 * error_train
	@printf "%.2f %% on test set\n" 100 * error_test

	return error_train, error_test
end

function printerrorsAE(net, smallimgs, smallimgstest, conf)
	error_train = geterrorsAE(net, smallimgs, nonlinearity = conf.nonlinearity)
	error_test = geterrorsAE(net, smallimgstest, nonlinearity = conf.nonlinearity)
	@printf "%.f on training set\n" error_train
	@printf "%.f on test set\n" error_test

	return error_train, error_test
end

#unlablled data
function printerrorsAE(net, smallimgs, conf)
	error_train = geterrorsAE(net, smallimgs, nonlinearity = conf.nonlinearity)
	@printf "%.f on training set\n" error_train

	return error_train
end

#####################################################
#Distortion functions (bring in noise)

function GaussianNoiseOnInput!(net,noise_amplitude::Float64)
	#additive Gaussian noise
	net.x[1] += noise_amplitude*randn(length(net.x[1]))
end

function MaskingNoiseOnInput!(net,noise_fraction::Float64)
	if noise_fraction < 0
		print("noise fraction out of bounds (<0)! should be in [0,1]")
	elseif  noise_fraction > 1
		print("noise fraction out of bounds (>1)! should be in [0,1]")
	end
	#force fraction "noise_fraction" of the input units to 0
	net.x[1][rand(range(1,length(net.x[1])),Int(floor(noise_fraction*length(net.x[1]))))] = 0.
end

function SaltPepperNoiseOnInput!(net,noise_fraction::Float64)
	if noise_fraction < 0
		print("noise fraction out of bounds (<0)! should be in [0,1]")
	elseif  noise_fraction > 1
		print("noise fraction out of bounds (>1)! should be in [0,1]")
	end
	#force fraction "noise_fraction" of the input units to either max or min value of inputs
	net.x[1][rand(range(1,length(net.x[1])),Int(floor(noise_fraction*length(net.x[1]))))]	= rand([minimum(net.x[1]),maximum(net.x[1])],Int(floor(noise_fraction*length(net.x[1]))))
end

function ChooseNoise(noise_type::String)
	if noise_type == "Gaussian"
		return GaussianNoiseOnInput!
	elseif noise_type == "Masking"
		return MaskingNoiseOnInput!
	elseif noise_type == "SaltPepper"
		return SaltPepperNoiseOnInput!
	end
end


#####################################################
# Transform SAE to network (for later fine-tuning e.g.)

function SAEtoNet(inputsize,outputsize,sae::SAE)
	#construct array of hidden layer sizes
	ns = vcat(inputsize,[size(sae.reps[i][1])[1] for i in 1:sae.nae],outputsize)
	#create net as deep classifier
	net = Classifier(ns)
	for i in 1:length(ns)-2 #last layer stays randomly initialized
		net.w[i] = deepcopy(sae.aes[i].w[1])
		net.b[i] = deepcopy(sae.aes[i].b[1])
 	end
	return net
end

function SAE_sparse_toNet(sae::SAE_sparse)
	#construct array of hidden layer sizes
	ns = vcat(144,[size(sae.reps[i][1])[1] for i in 1:sae.nae],10)
	#create net as deep classifier
	net = AE_sparse(ns)
	for i in 1:length(ns)-2 #last layer stays randomly initialized
		net.w[i] = deepcopy(sae.aes[i].w[1])
		net.b[i] = deepcopy(sae.aes[i].b[1])
 	end
	return net
end
#ATTENTION: THIS NETWORK SHOULD NOT BE USED FOR ORDINARY FINE-TUNING!
#SPECIAL FORWARD PROP NEEDED!!!

#####################################################
# Generative sampling from stacked autoencoder

function BernoulliSampling(y::Array{Float64, 1})

end

function backwardprop!(net::SAE)

end

#Generates nsampl samples (input representations) using trained SAE as generative model.
#1. Generate top-layer repr. by forwardprop of ONE randomly picked data sample
#2. alternating BernoulliSampling and
#		top-down inference (decoding path)
function GenerateSample(smallimgs::Array{Float64, 2},
				net::SAE,layer::Int64,nsampl::Int64)
	inputsample = smallimgs[:,rand(1:size(smallimgs)[2])]
end
