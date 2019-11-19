#############################################################
#Data Import and Preprocesing

using HDF5

## Helpers and Preprocesing

function subtractmean!(data)
        m = mean(data, dims=2)
        d, n = size(data)
        for i in 0:n-1
                BLAS.axpy!(-1., m, 1:d, data, i*d + 1: (i+1) * d)
        end
end
export subtractmean!
function subtractmean(data)
        m = deepcopy(data)
        subtractmean!(m)
        m
end
export subtractmean

#scale data between [-1,1]
function rescaledata!(data)
        absmax = maximum(abs(data))
        scale!(data,1. / absmax)
end

# Johanni's implementation
# data is dxn array of n d-dimensional random vectors with mean 0
function whiten(data; method = :f_ZCA)
        f = svd(data)
        eval(method)(f) * sqrt(size(data, 2) - 1)
end

f_ZCA(f::SVD) = f.U * f.Vt
f_PCA(f::SVD) = f.Vt

using Statistics
function downsample(data; factor = 3)
    imgsize = size(data)
    n_imgsize = Int(imgsize[1] / factor)
    n_imgs = zeros(n_imgsize, n_imgsize, imgsize[3])
    for p in 1:imgsize[3]
        for i in 1:n_imgsize
            for j in 1:n_imgsize
                n_imgs[i, j, p] = mean(data[factor*(i-1)+1:factor*i, factor*(j-1)+1:factor*j, p])
            end
        end
    end
    n_imgs
end
export downsample

function crop(data; margin = 15)
    imgsize = size(data)
    n_imgsize = imgsize[1] - 2 * margin
    (2 * margin >= imgsize[1]) && error("margin bigger than image!")
    n_imgs = zeros(n_imgsize, n_imgsize, imgsize[3])
    for i in 1:imgsize[3]
        n_imgs[:,:,i] = data[margin+1:end-margin, margin+1:end-margin, i]
    end
    n_imgs
end
export crop

function zeropad(smallimgs; targetsize = 32)
	n_imgs = size(smallimgs)[2]
	insize = Int(sqrt(size(smallimgs)[1]))
	smallimgs = reshape(smallimgs, insize, insize, n_imgs)
	imgs = zeros(targetsize, targetsize, n_imgs)

	margin = Int(floor((targetsize - insize) / 2))
	for i in 1:n_imgs
		imgs[margin:margin+insize-1, margin:margin+insize-1, i] = smallimgs[:, :, i]
	end
	return imgs
end
export zeropad


## Data imports
#
# function import_data(data::String)
#     if data == "MNIST"
#         datastring = "GitLab_exercise/mnist.mat"
# 	elseif data == "MNIST_shifted"
# 		datastring = "GitLab_exercise/MNIST_shifted.mat"
#     elseif data == "MNIST144"
#         datastring = "GitLab_exercise/mnist144.mat"
#     elseif data == "MNIST144_white"
#         datastring = "GitLab_exercise/MNIST144_whiteZCA.mat"
#     elseif data == "MNIST144_shifted"
# 		datastring = "GitLab_exercise/MNIST144_shifted.mat"
#     elseif data == "CIFAR10"
#         datastring = "cifar-10-batches-py/CIFAR10_all.mat"
#     elseif data == "CIFAR10_whitened"
#         datastring = "cifar-10-batches-py/CIFAR10_all_ZCA_white.mat"
#     end
#
#
#   if Sys.isapple()
#     path = "/Users/Bernd/Documents/PhD/Projects/"
#   elseif Sys.islinux()
#     path = "/home/illing/"
#   end
#
#   print("using matlab import...\n")
# 	print(string("load: ",data,"\n"))
#   if data == "CIFAR10"
# 	file = matopen(string(path,datastring))
#   elseif data == "CIFAR10_whitened"
#     file = h5open(string(path,datastring))
#   else
# 	file = h5open(string(path,datastring))
#   end
#   smallimgs = read(file, "trainingimages")
#   labels = read(file, "traininglabels")
#   smallimgstest = read(file, "testimages");
#   labelstest =  read(file, "testlabels");
#   close(file)
#
# 	if data == "CIFAR10"
# 		smallimgs = convert(Array{Float64, 2}, reshape(smallimgs,50000,32*32)')
# 		labels = convert(Array{Float64, 1},reshape(labels,50000))
# 		smallimgstest = convert(Array{Float64, 2}, reshape(smallimgstest,10000,32*32)')
# 		labelstest = convert(Array{Float64, 1},reshape(labelstest,10000))
# 	end
# 	n_trainsamples = size(smallimgs)[2]
# 	n_testsamples = size(smallimgstest)[2]
#
#     data_max = maximum([maximum(abs.(smallimgs)),maximum(abs.(smallimgstest))])
#     smallimgs ./= data_max
#     smallimgstest ./= data_max
#   return smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples
# end
# export import_data
#
# # using Knet
# # include(Knet.dir("data", "cifar.jl"))
# # include(Knet.dir("data", "mnist.jl"))
# # function import_data_Knet(dataset::String)
# # 	if dataset == "CIFAR10"
# # 		smallimgs, labels, smallimgstest, labelstest = cifar10()
# # 	    smallimgs = reshape(smallimgs, 32^2 * 3, 50000)
# # 	    smallimgstest = reshape(smallimgstest, 32^2 * 3, 10000)
# # 	elseif dataset == "MNIST"
# # 		smallimgs, labels, smallimgstest, labelstest = mnist()
# # 	    smallimgs = reshape(smallimgs, 28^2, 60000)
# # 	    smallimgstest = reshape(smallimgstest, 28^2, 10000)
# # 	end
# # 	smallimgs, labels .- 1, smallimgstest, labelstest .- 1, size(smallimgs)[2], size(smallimgstest)[2]
# # end
# # export import_data_Knet
#
# function import_unlabelled_data(data::String)
#   if Sys.isapple()
#     path = "/Users/Bernd/Documents/PhD/Projects/"
#   elseif Sys.islinux()
#     path = "/home/illing/"
#   end
#
# 	if data == "Olshausen"
# 		datastring = "natural_images/patchesOlshausen.jld"
#     print(string("load: ",data,"\n"))
#   	smallimgs = load(string(path,datastring),"patches")
#     n_samples = size(smallimgs)[2]
#   elseif data == "Olshausen_white"
#   	datastring = "natural_images/patchesOlshausen.jld"
#     print(string("load: ",data,"\n"))
#   	smallimgs = load(string(path,datastring),"wpatches")
#     n_samples = size(smallimgs)[2]
# 	elseif data == "VanHateren"
# 		datastring = "natural_images/patchesVanHateren.jld"
#     print(string("load: ",data,"\n"))
#   	smallimgs = load(string(path,datastring),"patches")
#     n_samples = size(smallimgs)[2]
#   elseif data == "VanHateren_white"
# 		datastring = "natural_images/patchesVanHateren.jld"
#     print(string("load: ",data,"\n"))
#   	smallimgs = load(string(path,datastring),"wpatches")
#     n_samples = size(smallimgs)[2]
#   elseif data == "bars"
#     datastring = "SparsePooling/artificial_data/moving_bars/all_bars.jld"
#     print(string("load: ",data,"\n"))
#   	smallimgs = load(string(path,datastring),"data")
#     n_samples = size(smallimgs)[2]
#   elseif data == "bars_superimposed"
#     datastring = "SparsePooling/artificial_data/moving_bars/superimposed_bars.jld"
#     print(string("load: ",data,"\n"))
#   	smallimgs = load(string(path,datastring),"data")
#     n_samples = size(smallimgs)[2]
#   end
#
# 	return smallimgs, n_samples #n_testsamples = 0
# end

# Docs:
# https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/
# The training set is composed of 5 instances of each category (instances 4, 6, 7, 8 and 9),
# and the test set of the remaining 5 instances (instances 0, 1, 2, 3, and 5).
function import_smallNORB(datasplit::String) # "train" or "test"
    if Sys.isapple()
        path = "/Users/Bernd/Documents/PhD/Projects/natural_images/small_NORB/"
    elseif Sys.islinux()
        path = "/root/small_NORB/"
    end
    (datasplit == "train") ? (file = h5open(string(path,"data_train.h5"))) :
        (file = h5open(string(path,"data_test.h5")))
    images_lt = permutedims(convert(Array{Float64},read(file,"images_lt")),[2,1,3])
    images_rt = permutedims(convert(Array{Float64},read(file,"images_rt")),[2,1,3])
    category_list = convert(Array{Int64},read(file,"categories"))
    instance_list = read(file,"instances")
    elevation_list = convert(Array{Int64},read(file,"elevations"))
    azimuth_list = convert(Array{Int64},read(file,"azimuths"))
    lighting_list = convert(Array{Int64},read(file,"lightings"))

    return images_lt ./ maximum(images_lt), images_rt ./ maximum(images_rt),
        category_list, instance_list, elevation_list,
        azimuth_list, lighting_list, length(category_list)
end
export import_smallNORB


function getNORB()
    images_lt, images_rt, category_list, instance_list, elevation_list,
        azimuth_list, lighting_list = import_smallNORB("train");
    images_lt_test, images_rt_test, category_list_test, instance_list_test, elevation_list_test,
        azimuth_list_test, lighting_list_test = import_smallNORB("test");

    images = downsample(crop(images_lt; margin = 16); factor = 2);
    images_test = downsample(crop(images_lt_test; margin = 16); factor = 2);

    sz = size(images)
    images = reshape(subtractmean(reshape(images,sz[1]^2,sz[3])),sz[1],sz[2],sz[3])
    images_test = reshape(subtractmean(reshape(images_test,sz[1]^2,sz[3])),sz[1],sz[2],sz[3])

    data = NORBdata(images, category_list, instance_list, elevation_list, azimuth_list, lighting_list)
    datatest = NORBdata(images_test, category_list_test, instance_list_test, elevation_list_test, azimuth_list_test, lighting_list_test)

    ind = data.nsamples # 5000 # 50000 # for training & evaluating classifier
    ind_test = datatest.nsamples # 5000 # 10000

    return data, datatest, ind, ind_test
end
export getNORB


function getMNIST()
    if Sys.isapple()
        path = "/Users/Bernd/Documents/PhD/Projects/"
    elseif Sys.islinux()
        path = "/root/"
    end
    file = h5open(string(path,"mnist.mat"))
    smallimgs = read(file, "trainingimages")
    labels = read(file, "traininglabels")
    smallimgstest = read(file, "testimages");
    labelstest =  read(file, "testlabels");
    close(file)

    n_trainsamples = size(smallimgs)[2]
	n_testsamples = size(smallimgstest)[2]

    data_max = maximum([maximum(abs.(smallimgs)),maximum(abs.(smallimgstest))])
    smallimgs ./= data_max
    smallimgstest ./= data_max

    smallimgs = subtractmean(smallimgs)
    smallimgstest = subtractmean(smallimgstest)

    return smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples
end
export getMNIST

using StatsBase
function reduceMNIST(imgs, labels; nclasses = 10, nperclass = 20)
    Random.seed!(1234)
    reducedimgs = zeros(size(imgs, 1), nclasses * nperclass)
    reducedlabels = zeros(nclasses * nperclass)
    for class in 0:9
		classinds = findall(labels .== class)
		reducedclassinds = sample(classinds, nperclass, replace = false)
		reducedimgs[:, class*nperclass+1:(class+1)*nperclass] = imgs[:, reducedclassinds]
        reducedlabels[class*nperclass+1:(class+1)*nperclass] = labels[reducedclassinds]
    end
    return reducedimgs, reducedlabels, nclasses * nperclass
end
export reduceMNIST

function getPaddedMNIST(; targetsize = 40, margin = div(targetsize - 28, 2) + 3, reduce = false)
    smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples =
		getMNIST();

    if reduce
        smallimgs, labels, n_trainsamples = reduceMNIST(smallimgs, labels)
        smallimgstest, labelstest, n_testsamples = reduceMNIST(smallimgstest, labelstest)
    end

	imgs = reshape(zeropad(smallimgs; targetsize = targetsize), targetsize^2, n_trainsamples)
	imgstest = reshape(zeropad(smallimgstest; targetsize = targetsize), targetsize^2, n_testsamples)

	data = labelleddata(imgs, labels; margin = margin)
	datatest = labelleddata(imgstest, labelstest; margin = margin)

    ind = data.nsamples
    ind_test = datatest.nsamples

    return data, datatest, ind, ind_test
end
export getPaddedMNIST

using Metalhead
getarray(X) = Float64.(permutedims(channelview(X), (2, 3, 1)))
function reformatimgs(data, length)
    labels = [data[i].ground_truth.class for i in 1:length]
    imgs = zeros(32*32, 3, length)
    for i in 1:length imgs[:, :, i] = reshape(getarray(data[i].img), 32*32, 3) end
    return imgs, labels
end
function getCIFAR10(; greyscale = false)
    @info("Loading data set: CIFAR10 (color)")
    if greyscale
        datastring = "CIFAR10_all.mat"
        if Sys.isapple()
          path = "/Users/Bernd/Documents/PhD/Projects/cifar-10-batches-py/"
        elseif Sys.islinux()
          path = "/home/illing/"
        end
        file = matopen(string(path,datastring))
        imgs = convert(Array{Float64, 2}, reshape(read(file, "trainingimages"),50000,32*32)')
 		labels = convert(Array{Float64, 1},reshape(read(file, "traininglabels"),50000))
 		imgstest = convert(Array{Float64, 2}, reshape(read(file, "testimages"),10000,32*32)')
 		labelstest = convert(Array{Float64, 1},reshape(read(file, "testlabels"),10000))
        close(file)
    else
        imgs, labels = reformatimgs(trainimgs(CIFAR10), 50000)
        imgstest, labelstest = reformatimgs(valimgs(CIFAR10), 10000)
    end
    data = labelleddata(imgs, labels)
	datatest = labelleddata(imgstest, labelstest)

    return data, datatest
end
export getCIFAR10
