using Pkg; Pkg.activate("./../SparsePooling/")#; Pkg.instantiate()
push!(LOAD_PATH, "./../SparsePooling/src/")
using SparsePooling
using PyPlot

mode =  "SparsePooling" #"MLP" #"SP" #

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
ind_t = datatest.nsamples # 5000 # 10000


## SparsePooling network
if mode == "SparsePooling"
    network = net(["input","sparse_patchy","pool_patchy","sparse_patchy"],
                [size(data.data)[1]^2,10,10,20],
                [0,6,3,4], #0,6,3
                [0,2,1,2]) #stride 1 in first layer works best so far

    ## Training
    inputfunction = getsmallimg
    intermediatestates = []
    learn_net_layerwise!(network, data, intermediatestates,
        [10^4,10^3,10^4,10^3],
        [inputfunction for i in 1:network.nr_layers],
        [getstaticimage, getmovingimage,getstaticimage,getmovingimage];
        LearningFromLayer = 2,
        LearningUntilLayer = network.nr_layers)

    #TODO patchy input for NORB data type for weight sharing!

    lasthiddenrepstrain = labelleddata(generatehiddenreps!(network, data;
            ind = ind, normalize = true, subtractmean = false), data.labels[1:ind]; classes = 0:4)
    lasthiddenrepstest = labelleddata(generatehiddenreps!(network, datatest;
            ind = ind_t, normalize = true, subtractmean = false), datatest.labels[1:ind_t]; classes = 0:4)
    traintopendclassifier!(network, lasthiddenrepstrain, lasthiddenrepstest; hidden_sizes = Int64[],
                iters = 10^6, ind = ind, indtest = ind_t, n_classes = 5)

## MLP control
elseif mode == "MLP"
    traintopendclassifier!(net(["input"],[size(data.data)[1]^2],[0],[0]), data, datatest; hidden_sizes = [128],
			iters = 10^6, ind = data.nsamples, indtest = datatest.nsamples,
			n_classes = 5, inputfunction = getsmallimg, movingfunction = getstaticimage)
elseif mode == "SP"
    traintopendclassifier!(net(["input"],[size(data.data)[1]^2],[0],[0]), data, datatest; hidden_sizes = Int64[],
			iters = 10^6, ind = data.nsamples, indtest = datatest.nsamples,
			n_classes = 5, inputfunction = getsmallimg, movingfunction = getstaticimage)
end

## Plotting of the NORB sequences

# getsmallimg(data)
#
# duration = 20
# seq = get_sequence_smallNORB(images, category_list, instance_list,
#         elevation_list, azimuth_list, lighting_list;
#         duration = duration, move = "rotate_horiz");
#
# for i in 1:duration
#     imshow(seq[:, :, i], cmap = "gray")
#     sleep(0.2)
# end
