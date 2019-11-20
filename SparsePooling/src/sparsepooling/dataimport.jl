#############################################################
#Data Import and Preprocesing

using HDF5
using MLDatasets

## Helpers and Preprocesing
function subtractmean!(data::Array{Float64, 2})
        m = mean(data, dims=2)
        d, n = size(data)
        for i in 0:n-1
                BLAS.axpy!(-1., m, 1:d, data, i*d + 1: (i+1) * d)
        end
end
export subtractmean!
function subtractmean!(data::Array{Float64, 3})
    s = size(data)
    subtractmean!(reshape(data, s[1] * s[2], s[3]))
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
    @info("Loading data set: small NORB")
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


function loadMNIST()
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

function getMNIST()
    @info("Loading data set: MNIST")
    imgs, labels, imgstest, labelstest, n_trainsamples, n_testsamples =
		loadMNIST();
    data = labelleddata(imgs, labels)
	datatest = labelleddata(imgstest, labelstest)

    return data, datatest, data.nsamples, datatest.nsamples
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
    @info("Loading data set: Padded MNIST (for shifted MNIST)")
    smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples =
		loadMNIST();

    if reduce
        smallimgs, labels, n_trainsamples = reduceMNIST(smallimgs, labels)
        smallimgstest, labelstest, n_testsamples = reduceMNIST(smallimgstest, labelstest)
    end

	imgs = reshape(zeropad(smallimgs; targetsize = targetsize), targetsize^2, n_trainsamples)
	imgstest = reshape(zeropad(smallimgstest; targetsize = targetsize), targetsize^2, n_testsamples)

	data = labelleddata(imgs, labels; margin = margin)
	datatest = labelleddata(imgstest, labelstest; margin = margin)

    return data, datatest, data.nsamples, datatest.nsamples
end
export getPaddedMNIST

function reformatimgs((x, y))
    imgs = convert.(Float64, x)
    labels = float.(y)
    imgs = reshape(convert.(Float64, x), 32 * 32, 3, size(x)[end])
    return imgs, labels
end
function getCIFAR10(; greyscale = false)
    if greyscale
        @info("Loading data set: CIFAR10 (greyscale)")
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
        @info("Loading data set: CIFAR10 (color)")
        imgs, labels = reformatimgs(CIFAR10.traindata())
        imgstest, labelstest = reformatimgs(CIFAR10.testdata())
    end

    data_max = maximum([maximum(abs.(imgs)),maximum(abs.(imgstest))])
    imgs ./= data_max
    imgstest ./= data_max

    imgs = subtractmean(imgs)
    imgstest = subtractmean(imgstest)

    data = labelleddata(imgs, labels; color = !greyscale)
	datatest = labelleddata(imgstest, labelstest; color = !greyscale)

    return data, datatest, data.nsamples, datatest.nsamples
end
export getCIFAR10
