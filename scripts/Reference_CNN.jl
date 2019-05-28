using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Metalhead
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using LinearAlgebra, ProgressMeter, JLD2, FileIO, MAT, Random
#using CuArrays # ATTENTION: This decides whether GPU or CPU is used!!!
include("./../sparsepooling/dataimport.jl")

data_set = "CIFAR10_gray" #"CIFAR10_gray" # "MNIST" #
epochs = 20
batch_size = 128 # 500
index = 10000 # for evaluating accuracy
n_in_channel = (data_set == "CIFAR10") ? 3 : 1

if data_set == "CIFAR10"
    @info("Loading data set: CIFAR10 (color)")
    getarray(X) = Float64.(permutedims(channelview(X), (2, 3, 1)))

    X = trainimgs(CIFAR10)
    imgs = [getarray(X[i].img) for i in 1:50000]
    labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, batch_size)]) |> gpu
    X_all = zeros(32,32,3,50000)
    for i in 1:50000 X_all[:,:,:,i] = imgs[i] end

    X_test = valimgs(CIFAR10)
    testimgs = [getarray(X_test[i].img) for i in 1:10000]
    testlabels = onehotbatch([X_test[i].ground_truth.class for i in 1:10000], 1:10) |> gpu
    testX = zeros(32,32,3,10000)
    for i in 1:10000 testX[:,:,:,i] = testimgs[i] end

    valset = collect(49001:50000)
    valX = cat(imgs[valset]..., dims = 4) |> gpu
    vallabels = labels[:, valset] |> gpu
elseif data_set == "CIFAR10_gray"
    @info("Loading data set: CIFAR10 (grayscale)")
    smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples = import_data("CIFAR10")

    imgs = [reshape(smallimgs[:,i],32,32,1) for i in 1:50000]
    labels = onehotbatch([labels[i] + 1 for i in 1:50000],1:10)
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, batch_size)])
    X_all = zeros(32,32,1,50000)
    for i in 1:50000 X_all[:,:,:,i] = imgs[i] end

    imgstest = [reshape(smallimgstest[:,i],32,32,1) for i in 1:10000]
    testlabels = onehotbatch([labelstest[i] + 1 for i in 1:10000],1:10)
    testX = zeros(32,32,1,10000)
    for i in 1:10000 testX[:,:,:,i] = imgstest[i] end

    valset = collect(49001:50000)
    valX = cat(imgs[valset]..., dims = 4) |> gpu
    vallabels = labels[:, valset] |> gpu
elseif data_set == "MNIST"
    @info("Loading data set: MNIST")
    imgs = MNIST.images(:train)
    imgstest = MNIST.images(:test)
    labels = gpu.(onehotbatch(MNIST.labels(:train), 0:9))
    labelstest = gpu.(onehotbatch(MNIST.labels(:test), 0:9))

    train = gpu.([(cat(float.(imgs[i])..., dims = 4), labels[:,i])
             for i in partition(1:50000, batch_size)])

    # Prepare test set (first 1,000 images) & for on-line testing
    X_all_temp = zeros(28,28,1,60000)
    for i in 1:60000 X_all_temp[:,:,:,i] = imgs[i] end
    X_all = X_all_temp |> gpu
    labels = labels[:,1:end] |> gpu

    testX_temp = zeros(28,28,1,10000)
    for i in 1:10000 testX_temp[:,:,:,i] = imgstest[i] end
    testX = testX_temp |> gpu
    testlabels = labelstest[:,1:end] |> gpu

    valset = collect(59001:60000)
    valX = cat(float.(imgs[valset])..., dims = 4) |> gpu
    vallabels = labels[:, valset] |> gpu
end

######################################################################

@info("Build CNN...")

Simple_CNN() = Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), n_in_channel => 32, stride=(1, 1), relu),
    #BatchNorm(32),
    x -> maxpool(x, (2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 32 => 64, stride=(1, 1), relu),
    #BatchNorm(64),
    x -> maxpool(x, (2,2)),

    # Third convolution, operating upon a 7x7 image
    #Conv((3, 3), 32 => 32, pad=(1,1), stride=(1, 1), relu),
    #BatchNorm(32),
    #x -> maxpool(x, (2,2)),

    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
    # which is where we get the 288 in the `Dense` layer below:
    Dropout(0.25),

    x -> reshape(x, :, size(x, 4)),
    (data_set == "MNIST") ? Dense(1600, 128, relu) : Dense(2304, 128, relu),
    Dropout(0.5),
    Dense(128, 10),
    softmax) |> gpu
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
  softmax) |> gpu

m = Simple_CNN()

loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))

# Defining the callback and the optimizer
evalcb = throttle(() -> @show(accuracy(valX, valY)), 10)

opt = ADAM()

@info("Train CNN...")
for i in 1:epochs
    @info(string("Epoch nr. ",i," out of ",epochs))
    @time Flux.train!(loss, params(m), train, opt)
    GC.gc()
    println("val acc: ", accuracy(valX, vallabels))
end

# Evaluate train and test accuracy
@info("Evaluate accuracies...")
GC.gc()
println("acc train: ", accuracy(X_all, labels))
GC.gc()
println("acc test: ", accuracy(testX, testlabels))
