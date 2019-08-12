# Script for data-augmentation of MNIST to enlarged and shifted digits
# and for making movies of moving MNIST digts

using HDF5, JLD2, FileIO, GR, ProgressMeter, LinearAlgebra
path = "./../../SparsePooling/src/sparsepooling"
include("$path/types.jl")
include("$path/helpers.jl")

function getMNIST(; path = "/Users/illing/")
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
  return smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples
end

function zeropad(smallimgs; targetsize = 32)
	n_imgs = size(smallimgs)[2]
	insize = Int(sqrt(size(smallimgs)[1]))
	smallimgs = reshape(smallimgs, insize, insize, n_imgs)
	imgs = zeros(targetsize, targetsize, n_imgs)

	margin = Int(floor((targetsize - insize) / 2))
	for i in 1:n_imgs
		imgs[margin:margin+insize-1, margin:margin+insize-1, i] = smallimgs[:, :, i]
	end
	return reshape(imgs, targetsize^2, n_imgs)
end

# functions for creating datasets for saving (for CNN reference)
function createshifteddataset(; targetsize = 50, margin = div(50 - 28, 2) + 3,
								duration_per_pattern = 1)
	smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples =
		getMNIST();
	newimgstrain = zeros(targetsize^2, duration_per_pattern * n_trainsamples)
	newimgstest = zeros(targetsize^2, duration_per_pattern * n_testsamples)
	newlabelstrain = zeros(duration_per_pattern * n_trainsamples)
	newlabelstest = zeros(duration_per_pattern * n_testsamples)

	for (oldimgs, oldlabels, newimgs, newlabels, old_n_samples) in
		[(smallimgs, labels, newimgstrain, newlabelstrain, n_trainsamples),
		(smallimgstest, labelstest, newimgstest, newlabelstest, n_testsamples)]

		imgs = zeropad(oldimgs; targetsize = targetsize)
		#imgstest = zeropad(smallimgstest; targetsize = targetsize)

		data = labelleddata(imgs, labels, margin)
		#floatMNISTdatatest = labelleddata(imgstest, labelstest, margin)

		for i in 1:old_n_samples
			indices = (i-1)*duration_per_pattern+1:i*duration_per_pattern
			newimgs[:, indices] =
				getmovingimage(data, data.data[:, i]; duration = duration_per_pattern)[:]
			newlabels[indices] .= oldlabels[i]
		end
	end
	save(string(pwd(),"/MNISTshifted.jld2"), "trainingimages", newimgstrain,
								"traininglabels", newlabelstrain,
								"testimages", newimgstest,
								"testlabels", newlabelstest)
end

### Testing

targetsize = 50
margin = div(50 - 28, 2) + 3
duration_per_pattern = margin

createshifteddataset(; targetsize = targetsize, margin = margin,
								duration_per_pattern = 1)

#
# smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples =
# 	getMNIST();
#
# imgs = zeropad(smallimgs; targetsize = targetsize)
# imgstest = zeropad(smallimgstest; targetsize = targetsize)
#
# floatMNISTdatatrain = labelleddata(imgs, labels, margin)
# floatMNISTdatatest = labelleddata(imgstest, labelstest, margin)
#
# for i in 1:50
# 	movimg = getmovingimage(floatMNISTdatatrain,
# 							floatMNISTdatatrain.data[:,rand(1:n_trainsamples)];
# 							duration = duration_per_pattern)
#
# 	for i in 1:duration_per_pattern
# 	  imshow(movimg[:,:,i]')
# 	  sleep(0.1)
# 	end
# 	sleep(0.5)
# end
