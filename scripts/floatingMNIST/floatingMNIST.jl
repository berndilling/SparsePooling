# Script for data-augmentation of MNIST to enlarged and shifted digits
# and for making movies of moving MNIST digts

using Pkg; Pkg.activate("./../../SparsePooling/"); Pkg.instantiate()
push!(LOAD_PATH, "./../../SparsePooling/src/")
using SparsePooling

using HDF5
using JLD2, FileIO, ProgressMeter, LinearAlgebra, GR, StatsBase
using Random

Random.seed!(1234)

# functions for creating datasets for saving (for CNN reference)
function createpaddedandshifteddata(; zero_pad = true, targetsize = 40, margin = div(targetsize - 28, 2) + 3,
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

		if zero_pad
			imgs = reshape(zeropad(oldimgs; targetsize = targetsize), targetsize^2, old_n_samples)
		end
		data = labelleddata(imgs, labels; margin = margin)

		for i in 1:old_n_samples
			indices = (i-1)*duration_per_pattern+1:i*duration_per_pattern
			newimgs[:, indices] =
				getmovingimage(data, data.data[:, i]; duration = duration_per_pattern)[:]
			newlabels[indices] .= oldlabels[i]
		end
		# TODO scramble order if duration_per_pattern > 1
	end
	save(string(pwd(),"/MNISTshifted.jld2"), "trainingimages", newimgstrain,
								"traininglabels", newlabelstrain,
								"testimages", newimgstest,
								"testlabels", newlabelstest)
end

function _shiftimage(img, amplitude)
	img_s = Int(sqrt(size(img, 1)))
	circshift(reshape(img,img_s,img_s), amplitude)[:]
end
function createreducedpaddedandshifteddata(; zero_pad = true, targetsize = 40, nclasses = 10, nperclass = 20, margin = div(targetsize - 28, 2) + 3)
	smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples =
		getMNIST();
	smallimgs, labels, n_trainsamples = reduceMNIST(smallimgs, labels; nclasses = nclasses, nperclass = nperclass)
	smallimgstest, labelstest, n_testsamples = reduceMNIST(smallimgstest, labelstest; nclasses = nclasses, nperclass = nperclass)

	npossibleshifts = (2 * margin + 1) ^ 2
	nnewsamples =  npossibleshifts * nclasses * nperclass
	newimgstrain = zeros(targetsize^2, nnewsamples)
	newimgstest = zeros(targetsize^2, n_testsamples)
	newlabelstrain = zeros(nnewsamples)
	newlabelstest = labelstest

	# train set
	paddedimgs = reshape(zeropad(smallimgs; targetsize = targetsize), targetsize^2, n_trainsamples)
	count = 1
	for i in 1:n_trainsamples
		for x in -margin:margin
			for y in -margin:margin
				newimgstrain[:, count] = _shiftimage(paddedimgs[:, i], [x, y])
				newlabelstrain[count] = labels[i]
				count += 1
			end
		end
	end

	# TODO what to do with test set? only one shift per image?
	paddedimgstest = reshape(zeropad(smallimgstest; targetsize = targetsize), targetsize^2, n_testsamples)
	for i in 1:n_testsamples
		dir = rand(-margin:margin, 2)
		newimgstest[:, i] = _shiftimage(paddedimgstest[:, i], dir)[:]
	end

	# shuffle order (even though this should be done by input function, e.g. getsmallimg)
	shuffledinds = sample(1:size(newimgstrain, 2), size(newimgstrain, 2); replace = false)
	newlabelstrain = newlabelstrain[shuffledinds]
	newimgstrain = [:, shuffledinds]

	# save dataset
	save(string(pwd(),"/MNISTreducedshifted.jld2"), "trainingimages", newimgstrain,
								"traininglabels", newlabelstrain,
								"testimages", newimgstest,
								"testlabels", newlabelstest)
end


targetsize = 40
margin = div(targetsize - 28, 2) + 3

#createpaddedandshifteddata(; targetsize = targetsize, margin = margin, duration_per_pattern = 1)

createreducedpaddedandshifteddata(; targetsize = targetsize, nperclass = 20, margin = margin)

#########################################################################
## Further testing

# duration_per_pattern = margin
#
# smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples =
# 	getMNIST();
#
# imgs = reshape(zeropad(smallimgs; targetsize = targetsize), targetsize^2, n_trainsamples)
# imgstest = reshape(zeropad(smallimgstest; targetsize = targetsize), targetsize^2, n_testsamples)
#
# floatMNISTdatatrain = labelleddata(imgs, labels; margin = margin)
# floatMNISTdatatest = labelleddata(imgstest, labelstest; margin = margin)
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
