
#############################################################
#Data Import and Preprocesing

function import_data(data::String)
  if data == "MNIST"
    datastring = "GitLab_exercise/mnist.mat"
	elseif data == "MNIST_shifted"
		datastring = "GitLab_exercise/MNIST_shifted.mat"
  elseif data == "MNIST144"
    datastring = "GitLab_exercise/mnist144.mat"
  elseif data == "MNIST144_white"
    datastring = "GitLab_exercise/MNIST144_whiteZCA.mat"
	elseif data == "MNIST144_shifted"
		datastring = "GitLab_exercise/MNIST144_shifted.mat"
  elseif data == "CIFAR10"
    datastring = "cifar-10-batches-py/CIFAR10_all.mat"
  elseif data == "CIFAR10_whitened"
    datastring = "cifar-10-batches-py/CIFAR10_all_ZCA_white.mat"
	end

  if is_apple()
    path = "/Users/Bernd/Documents/PhD/Projects/"
  elseif is_linux()
    path = "/home/illing/"
  end

  print("using matlab import...\n")
	print(string("load: ",data,"\n"))
  if data == "CIFAR10"
		file = matopen(string(path,datastring))
  elseif data == "CIFAR10_whitened"
    file = h5open(string(path,datastring))
	else
		file = h5open(string(path,datastring))
	end
	smallimgs = read(file, "trainingimages")
  labels = read(file, "traininglabels")
  smallimgstest = read(file, "testimages");
  labelstest =  read(file, "testlabels");
  close(file)

	if data == "CIFAR10"
		smallimgs = reshape(smallimgs,50000,32*32)'
		labels = convert(Array{Float64, 1},reshape(labels,50000))
		smallimgstest = reshape(smallimgstest,10000,32*32)'
		labelstest = convert(Array{Float64, 1},reshape(labelstest,10000))
	end
	n_trainsamples = size(smallimgs)[2]
	n_testsamples = size(smallimgstest)[2]

  return smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples
end

function import_unlabelled_data(data::String)
  if is_apple()
    path = "/Users/Bernd/Documents/PhD/Projects/"
  elseif is_linux()
    path = "/home/illing/"
  end

	if data == "Olshausen"
		datastring = "natural_images/patchesOlshausen.jld"
    print(string("load: ",data,"\n"))
  	smallimgs = load(string(path,datastring),"patches")
    n_samples = size(smallimgs)[2]
  elseif data == "Olshausen_white"
  	datastring = "natural_images/patchesOlshausen.jld"
    print(string("load: ",data,"\n"))
  	smallimgs = load(string(path,datastring),"wpatches")
    n_samples = size(smallimgs)[2]
	elseif data == "VanHateren"
		datastring = "natural_images/patchesVanHateren.jld"
    print(string("load: ",data,"\n"))
  	smallimgs = load(string(path,datastring),"patches")
    n_samples = size(smallimgs)[2]
  elseif data == "VanHateren_white"
		datastring = "natural_images/patchesVanHateren.jld"
    print(string("load: ",data,"\n"))
  	smallimgs = load(string(path,datastring),"wpatches")
    n_samples = size(smallimgs)[2]
  end

	return smallimgs, n_samples #n_testsamples = 0
end

#(By Johanni) subtract linewise (pixel-wise) mean
function subtractmean!(data)
        m = mean(data, 2)
        d, n = size(data)
        for i in 0:n-1
                BLAS.axpy!(-1., m, 1:d, data, i*d + 1: (i+1) * d)
        end
end
function subtractmean(data)
        m = deepcopy(data)
        subtractmean!(m)
        m
end

#scale data between [-1,1]
function rescaledata!(data)
        absmax = maximum(abs(data))
        scale!(data,1./absmax)
end

# Johanni's implementation
# data is dxn array of n d-dimensional random vectors with mean 0
function whiten(data; method = :f_ZCA)
        f = svdfact(data)
        eval(method)(f) * sqrt(size(data, 2) - 1)
end

f_ZCA(f::Base.LinAlg.SVD) = f[:U] * f[:Vt]
f_PCA(f::Base.LinAlg.SVD) = f[:Vt]
