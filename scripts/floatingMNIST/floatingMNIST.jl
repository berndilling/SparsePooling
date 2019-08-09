# Script for data-augmentation of MNIST to enlarged and shifted digits
# and for making movies of moving MNIST digts

using HDF5, GR

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

## TODO Here comes the getimage and getmovingimage functions that allow for
# sequences without periodic boundary conditions
# Attention: should use the data structure of the SparsePooling framework!

# TODO Think about how to store static augmented dataset for reference CNN training

###

targetsize = 50

smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples =
	getMNIST();

imgs = zeropad(smallimgs; targetsize = targetsize)

GR.imshow(reshape(imgs[:,1],targetsize,targetsize)')
