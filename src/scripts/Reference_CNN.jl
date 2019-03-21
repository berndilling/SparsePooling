using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Metalhead
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using LinearAlgebra, ProgressMeter, JLD2, FileIO, PyPlot, MAT
include("./../sparsepooling/dataimport.jl")

data_set = "CIFAR10_gray"#"MNIST" #"CIFAR10_gray"# "MNIST"#
index = 500 # for evaluating accuracy

if data_set == "CIFAR10"
    @info("Loading data set: CIFAR10 (color)")
    getarray(X) = Float64.(permutedims(channelview(X), (2, 3, 1)))

    X = trainimgs(dataset(CIFAR10))
    imgs = [getarray(X[i].img) for i in 1:50000]
    labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:50000, 1000)])

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
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:50000, 1000)])

    X = zeros(32,32,1,index)
    tX = zeros(32,32,1,index)
    tX_o = zeros(32,32,1,1000)
    for i in 1:index X[:,:,:,i] = imgs[i] end
    Y = labels[1:index]
    temp1 = [reshape(smallimgstest[:,i],32,32,1) for i in 1:10000]
    for i in 1:index tX[:,:,:,i] = temp1[i] end
    tY = onehotbatch([labelstest[i] + 1 for i in 1:index],1:10)
    temp2 = [reshape(smallimgstest[:,i],32,32,1) for i in 1:1000]
    for i in 1:1000 tX_o[:,:,:,i] = temp2[i] end |> gpu
    tY_o = onehotbatch([labelstest[i] + 1 for i in 1:1000],1:10) |> gpu
elseif data_set == "MNIST"
    @info("Loading data set: MNIST")
    imgs = MNIST.images()
    labels = onehotbatch(MNIST.labels(), 0:9)

    # Partition into batches of size 1,000
    train = gpu.([(cat(float.(imgs[i])..., dims = 4), labels[:,i])
             for i in partition(1:60000, 1000)]) #1:60_000
    #repeated((X, Y), 200)

    # Prepare test set (first 1,000 images) & for on-line testing
    X = cat(float.(MNIST.images(:train)[1:5000])..., dims = 4) |> gpu
    Y = onehotbatch(MNIST.labels(:train)[1:5000], 0:9) |> gpu
    tX = cat(float.(MNIST.images(:test)[1:5000])..., dims = 4) |> gpu
    tY = onehotbatch(MNIST.labels(:test)[1:5000], 0:9) |> gpu
    tX_o = cat(float.(MNIST.images(:test)[1:1000])..., dims = 4) |> gpu
    tY_o = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> gpu
end

######################################################################

# TODO add big convnet like vgg16() here!
# TODO How to control training time/several epochs?

@info("Build CNN...")
n_in_channel = (data_set == "CIFAR10") ? 3 : 1
m = Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), n_in_channel => 16, pad=(1,1), stride=(1, 1), relu),
    x -> maxpool(x, (2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad=(1,1), stride=(1, 1), relu),
    x -> maxpool(x, (2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad=(1,1), stride=(1, 1), relu),
    x -> maxpool(x, (2,2)),

    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
    # which is where we get the 288 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    (data_set == "MNIST") ? Dense(288, 10) : Dense(512, 10),

    # Finally, softmax to get nice probabilities
    softmax,
)

# m(train[1][1])
loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
evalcb = throttle(() -> @show(accuracy(tX_o, tY_o)), 10)
opt = ADAM(params(m))
#
@info("Train CNN...")
Flux.train!(loss, train, opt, cb = evalcb)

# Evaluate train and test accuracy
@info("Evaluate accuracies...")
println("acc X,Y ", accuracy(X, Y))
println("acc tX, tY ", accuracy(tX, tY))
