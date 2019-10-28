using Pkg; Pkg.activate("./../SparsePooling/"); Pkg.instantiate()
push!(LOAD_PATH, "./../SparsePooling/src/")
using SparsePooling
#include("./../sparsepooling/dataimport.jl")

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
#using Metalhead
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using LinearAlgebra, ProgressMeter, JLD2, FileIO, MAT, Random
using BSON: @save

use_gpu = true # helper to easily switch between gpu/cpu
todevice(x) = use_gpu ? gpu(x) : x
use_gpu && using CuArrays # ATTENTION: This decides whether GPU or CPU is used!!!

nettype = "MLP" #"CNN" #"SP"
data_set = "floatingMNIST" #"floatingreducedMNIST" #"MNIST" # "CIFAR10_gray" #"CIFAR10_gray" # "NORB" #
epochs = 20
batch_size = 128 # 500
n_in_channel = (data_set == "CIFAR10") ? 3 : 1

function getonechanneldataset(X, Y, batchsize, imsize)
    dataset = []
    mb_idxs = partition(1:size(X)[end], batchsize)
    for idxs in mb_idxs
        if length(size(Y)) == 1
            push!(dataset, (reshape(X[:,idxs], imsize, imsize, 1, length(idxs)), Y[idxs]))
        else
            push!(dataset, (reshape(X[:,idxs], imsize, imsize, 1, length(idxs)), Y[:,idxs]))
        end
    end
    return dataset
end

# ATTENTION: Only tested for MNIST & floating/reduced MNIST!
# TODO: Callback gives OutOfMemoryError for floatingMNIST! -> fix that!
if data_set == "CIFAR10"
    @info("Loading data set: CIFAR10 (color)")
    getarray(X) = Float64.(permutedims(channelview(X), (2, 3, 1)))

    X = trainimgs(CIFAR10)
    imgs = [getarray(X[i].img) for i in 1:50000]
    labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
    train = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, batch_size)] |> todevice
    X_all = zeros(32,32,3,50000)
    for i in 1:50000 X_all[:,:,:,i] = imgs[i] end

    X_test = valimgs(CIFAR10)
    testimgs = [getarray(X_test[i].img) for i in 1:10000]
    testlabels = onehotbatch([X_test[i].ground_truth.class for i in 1:10000], 1:10) |> todevice
    testX = zeros(32,32,3,10000)
    for i in 1:10000 testX[:,:,:,i] = testimgs[i] end

    valset = collect(49001:50000)
    valX = cat(imgs[valset]..., dims = 4) |> todevice
    vallabels = labels[:, valset] |> todevice
elseif data_set == "CIFAR10_gray"
    @info("Loading data set: CIFAR10 (grayscale)")
    smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples = import_data("CIFAR10")

    imgs = [reshape(smallimgs[:,i],32,32,1) for i in 1:50000]
    labels = onehotbatch([labels[i] + 1 for i in 1:50000],1:10)
    train = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, batch_size)] |> todevice
    X_all = zeros(32,32,1,50000)
    for i in 1:50000 X_all[:,:,:,i] = imgs[i] end

    imgstest = [reshape(smallimgstest[:,i],32,32,1) for i in 1:10000]
    testlabels = onehotbatch([labelstest[i] + 1 for i in 1:10000],1:10)
    testX = zeros(32,32,1,10000)
    for i in 1:10000 testX[:,:,:,i] = imgstest[i] end

    valset = collect(49001:50000)
    valX = cat(imgs[valset]..., dims = 4) |> todevice
    vallabels = labels[:, valset] |> todevice
elseif data_set == "NORB"
    images_lt, images_rt, category_list, instance_list, elevation_list,
        azimuth_list, lighting_list = import_smallNORB("train");
    images_lt_test, images_rt_test, category_list_test, instance_list_test, elevation_list_test,
        azimuth_list_test, lighting_list_test = import_smallNORB("test");

    images = downsample(crop(images_lt; margin = 16); factor = 2);
    images_test = downsample(crop(images_lt_test; margin = 16); factor = 2);

    imgs = [reshape(images[:,:,i],32,32,1) for i in 1:length(category_list)]
    labels = onehotbatch([category_list[i] + 1 for i in 1:length(category_list)],1:5)
    train = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:length(category_list)-5000, batch_size)] |> todevice
    X_all = zeros(32,32,1,length(category_list))
    for i in 1:length(category_list) X_all[:,:,:,i] = imgs[i] end

    imgstest = [reshape(images_test[:,:,i],32,32,1) for i in 1:length(category_list_test)]
    testlabels = onehotbatch([category_list_test[i] + 1 for i in 1:length(category_list_test)],1:5)
    testX = zeros(32,32,1,length(category_list_test))
    for i in 1:length(category_list_test) testX[:,:,:,i] = imgstest[i] end

    valset = collect(length(category_list_test)-5000+1:length(category_list))
    valX = cat(imgs[valset]..., dims = 4) |> todevice
    vallabels = labels[:, valset] |> todevice
elseif data_set == "MNIST"
    @info("Loading data set: MNIST")
    imgs = MNIST.images(:train)
    labels = onehotbatch(MNIST.labels(:train), 0:9)

    train = [(cat(float.(imgs[i])..., dims = 4), labels[:,i])
             for i in partition(1:50000, batch_size)]
    train = todevice.(train)

    X_all_temp = zeros(28,28,1,60000)
    for i in 1:60000 X_all_temp[:,:,:,i] = imgs[i] end
    X_all = X_all_temp
    labels = labels[:,1:end]

    imgstest = MNIST.images(:test)
    testX_temp = zeros(28,28,1,10000)
    for i in 1:10000 testX_temp[:,:,:,i] = imgstest[i] end
    testX = testX_temp #|> todevice
    testlabels = onehotbatch(MNIST.labels(:test), 0:9)

    valset = collect(59001:60000)
    valX = cat(float.(imgs[valset])..., dims = 4) |> todevice
    vallabels = labels[:, valset] |> todevice
elseif data_set == "floatingMNIST" || data_set == "floatingreducedMNIST"
    if data_set == "floatingMNIST"
        @info("Loading data set: floatingMNIST")
        data = load("./floatingMNIST/MNISTshifted.jld2")
    elseif data_set == "floatingreducedMNIST"
        @info("Loading data set: floatingreducedMNIST")
        data = load("./floatingMNIST/MNISTreducedshifted.jld2")
    end
    X_all = data["trainingimages"]
    labels = onehotbatch(data["traininglabels"], 0:9)
    testX = data["testimages"]
    testlabels = onehotbatch(data["testlabels"], 0:9)

    train = getonechanneldataset(X_all[:,1:end-10000], labels[:,1:end-10000], batch_size, 40)
    train = todevice.(train)

    X_all = reshape(X_all, 40, 40, 1, size(X_all)[end])
    testX = reshape(testX, 40, 40, 1, size(testX)[end])
    valset = collect(size(X_all)[end]-9999:size(X_all)[end])
    valX = reshape(X_all[:, :, :, valset], 40, 40, 1, length(valset)) |> todevice
    vallabels = labels[:, valset] |> todevice
end

######################################################################

@info("Build CNN/MLP...")
Simple_Perceptron(; n_classes = 10, imsize = 28) = Chain(
    x -> reshape(x, :, size(x)[end]),
    Dense(imsize ^ 2, n_classes),
    softmax) |> todevice
Simple_MLP(; nhidden = 5000, n_classes = 10, imsize = 28) = Chain(
    x -> reshape(x, :, size(x)[end]),
    Dense(imsize ^ 2, nhidden, relu),
    Dense(nhidden, n_classes),
    softmax) |> todevice
Simple_CNN(; n_classes = 10, stride = 1) = Chain(
    Conv((3, 3), n_in_channel => 32, stride=(stride, stride), relu),
    # BatchNorm(32),
    MaxPool((2,2)), # default: stride = pool window
    Conv((3, 3), 32 => 64, stride=(stride, stride), relu),
    # BatchNorm(64),
    MaxPool((2,2)),
    Conv((3, 3), 64 => 128, stride=(stride, stride), relu), # , pad=(1,1)
    # BatchNorm(128),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    # Dropout(0.25),
    Dense(1152, n_classes),
    softmax) |> todevice
vgg16() = Chain(
  Conv((3, 3), n_in_channel => 64, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64),
  Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64),
  x -> maxpool(x, (2, 2)),
  Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128),
  Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128),
  x -> maxpool(x, (2,2)),
  Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  x -> maxpool(x, (2, 2)),
  Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  x -> maxpool(x, (2, 2)),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  x -> maxpool(x, (2, 2)),
  x -> reshape(x, :, size(x, 4)),
  (data_set == "MNIST") ? Dense(288, 4096, relu) : Dense(512, 4096, relu),
  Dropout(0.5),
  Dense(4096, 4096, relu),
  Dropout(0.5),
  Dense(4096, 10),
  softmax) |> todevice

if nettype == "CNN"
    m = Simple_CNN()
elseif nettype == "MLP"
    m = Simple_MLP(; imsize = size(X_all, 1))
elseif nettype == "SP"
    m = Simple_Perceptron(; imsize = size(X_all, 1))
end

loss(x, y) = Flux.crossentropy(m(x), y)
evalcb = throttle(() -> @show(loss(valX, vallabels)), 2)
accuracy(x, y; n_classes = 10) = mean(onecold(cpu(m)(x), 1:n_classes) .== onecold(y, 1:n_classes))
opt = ADAM()

# if only last layer should be learned, e.g. for RP: trainableparams =  params(m[end-1])
trainableparams = params(m) #params(m[end - 1]) #

@info("Train CNN/MLP...")
for i in 1:epochs
    @info(string("Epoch nr. ",i," out of ",epochs))
    @time Flux.train!(loss, trainableparams, train, opt)#; cb = evalcb)
    println("acc validation: ", accuracy(testX, testlabels; n_classes = size(labels)[1]))
end

# Evaluate train and test accuracy
@info("Evaluate accuracies...")
println("acc train: ", accuracy(X_all, labels; n_classes = size(labels)[1]))
println("acc test: ", accuracy(testX, testlabels; n_classes = size(labels)[1]))

referencenetwork = cpu(m)
@save string("./floatingMNIST/Reference_", nettype, "_", data_set,".bson") referencenetwork
