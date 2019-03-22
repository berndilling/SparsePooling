using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Metalhead
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using LinearAlgebra, ProgressMeter, JLD2, FileIO, PyPlot, MAT, Random
include("./../sparsepooling/dataimport.jl")

data_set =  "MNIST" #"CIFAR10_gray" # "CIFAR10" #
epochs = 10
batch_size = 1000 # 500
index = 10000 # for evaluating accuracy
n_in_channel = (data_set == "CIFAR10") ? 3 : 1

if data_set == "CIFAR10"
    @info("Loading data set: CIFAR10 (color)")
    getarray(X) = Float64.(permutedims(channelview(X), (2, 3, 1)))

    X = trainimgs(dataset(CIFAR10))
    imgs = [getarray(X[i].img) for i in 1:50000]
    labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:50000, batch_size)])

    val_X = valimgs(dataset(CIFAR10))
    valX = [getarray(val_X[i].img) for i in 1:10000] |> gpu
    valY = onehotbatch([val_X[i].ground_truth.class for i in 1:10000],1:10) |> gpu

    # Prepare test set (first 1,000 images) & for on-line testing
    X = zeros(32,32,3,index)
    tX = zeros(32,32,3,index)
    tX_o = zeros(32,32,3,1000)
    for i in 1:index X[:,:,:,i] = imgs[i] end
    Y = labels[1:index]
    for i in 1:index tX[:,:,:,i] = valX[i] end
    tY = valY[1:index]
    temp = [getarray(val_X[i].img) for i in 1:1000]
    for i in 1:1000 tX_o[:,:,:,i] = temp[i] end
    tY_o = onehotbatch([val_X[i].ground_truth.class for i in 1:1000],1:10)
elseif data_set == "CIFAR10_gray"
    @info("Loading data set: CIFAR10 (grayscale)")
    smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples = import_data("CIFAR10")

    imgs = [reshape(smallimgs[:,i],32,32,1) for i in 1:50000]
    labels = onehotbatch([labels[i] + 1 for i in 1:50000],1:10)
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:50000, batch_size)])

    X = zeros(32,32,1,index)
    tX = zeros(32,32,1,index)
    tX_o = zeros(32,32,1,1000)
    for i in 1:index X[:,:,:,i] = imgs[i] end
    Y = labels[:,1:index]
    temp1 = [reshape(smallimgstest[:,i],32,32,1) for i in 1:10000]
    for i in 1:index tX[:,:,:,i] = temp1[i] end
    tY = onehotbatch([labelstest[i] + 1 for i in 1:index],1:10)
    temp2 = [reshape(smallimgstest[:,i],32,32,1) for i in 1:1000]
    for i in 1:1000 tX_o[:,:,:,i] = temp2[i] end |> gpu
    tY_o = onehotbatch([labelstest[i] + 1 for i in 1:1000],1:10) |> gpu
elseif data_set == "MNIST"
    @info("Loading data set: MNIST")
    imgs = MNIST.images(:train)
    imgstest = MNIST.images(:test)
    labels = onehotbatch(MNIST.labels(:train), 0:9)
    labelstest = onehotbatch(MNIST.labels(:test), 0:9)

    # Partition into batches of size 1,000
    train = gpu.([(cat(float.(imgs[i])..., dims = 4), labels[:,i])
             for i in partition(1:60000, batch_size)])

    # Prepare test set (first 1,000 images) & for on-line testing
    X = cat(float.(MNIST.images(:train))[1:index]..., dims = 4) |> gpu
    Y = onehotbatch(MNIST.labels(:train)[1:index], 0:9) |> gpu
    tX = cat(float.(MNIST.images(:test))..., dims = 4) |> gpu
    tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu
    #tX_o = cat(float.(MNIST.images(:test)[1:1000])..., dims = 4) |> gpu
    #tY_o = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> gpu
end

######################################################################

@info("Build CNN...")

Simple_CNN() = Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), n_in_channel => 32, pad=(1,1), stride=(1, 1), relu),
    #BatchNorm(32),
    x -> maxpool(x, (2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 32 => 64, pad=(1,1), stride=(1, 1), relu),
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
    (data_set == "MNIST") ? Dense(3136, 128, relu) : Dense(4096, 128, relu),
    Dropout(0.5),
    Dense(128, 10, relu),
    softmax)
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

m = Simple_CNN() #vgg16() #

loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
#evalcb = throttle(() -> @show(accuracy(tX_o, tY_o)), 10)
#evalcb = () -> @show(accuracy(tX_o, tY_o))
opt = ADAM(params(m))

@info("Train CNN...")
for i in 1:epochs
    @info(string("Epoch nr. ",i," out of ",epochs))
    @time Flux.train!(loss, train[randperm(length(train))], opt)#, cb = evalcb)
    @show(accuracy(X, Y))
    @show(accuracy(tX, tY))
end

# Evaluate train and test accuracy
@info("Evaluate accuracies...")
println("acc X,Y ", accuracy(X, Y))
println("acc tX, tY ", accuracy(tX, tY))
