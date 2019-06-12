using Pkg; Pkg.activate("./../SparsePooling/")#; Pkg.instantiate()
push!(LOAD_PATH, "./../SparsePooling/src/")
using SparsePooling
using PyPlot

mode = "SP" # "SparsePooling" #"MLP" #

images_lt, images_rt, category_list, instance_list, elevation_list,
    azimuth_list, lighting_list = import_smallNORB("train");
images_lt_test, images_rt_test, category_list_test, instance_list_test, elevation_list_test,
    azimuth_list_test, lighting_list_test = import_smallNORB("test");

images = downsample(crop(images_lt; margin = 16); factor = 2);
images_test = downsample(crop(images_lt_test; margin = 16); factor = 2);

data = NORBdata(images, category_list, instance_list, elevation_list, azimuth_list, lighting_list)
datatest = NORBdata(images_test, category_list_test, instance_list_test, elevation_list_test, azimuth_list_test, lighting_list_test)

## SparsePooling network
if mode == "SparsePooling"
    ind = 5000 # 50000 # for training & evaluating classifier
    ind_t = 5000 # 10000

    network = net(["input","sparse_patchy","pool_patchy"],
                [size(data.data)[1]^2,10,10],
                [0,6,3],
                [0,1,2])

    ## Training
    inputfunction = getsmallimg
    intermediatestates = []
    learn_net_layerwise!(network, data, intermediatestates,
        [10^4,10^2],
        [inputfunction for i in 1:network.nr_layers],
        [getstaticimage, getmovingimage];
        LearningFromLayer = 2,
        LearningUntilLayer = network.nr_layers)

    #TODO patchy input for NORB data type for weight sharing!

    lasthiddenrepstrain = labelleddata(generatehiddenreps!(network, data;
            ind = ind, normalize = true, subtractmean = false), data.labels[1:ind]; classes = 0:4)#,
            #instance_list, elevation_list, azimuth_list, lighting_list)
    lasthiddenrepstest = labelleddata(generatehiddenreps!(network, datatest;
            ind = ind_t, normalize = true, subtractmean = false), datatest.labels[1:ind_t]; classes = 0:4)#,
            #instance_list_test, elevation_list_test, azimuth_list_test, lighting_list_test)
    traintopendclassifier!(network, lasthiddenrepstrain, lasthiddenrepstest;
                iters = 10^6, ind = ind, indtest = ind_t, n_classes = 5)

## MLP control
elseif mode == "MLP"
    traintopendclassifier!(net(["input"],[size(data.data)[1]^2],[0],[0]), data, datatest; hidden_sizes = [500],
			iters = 10^5, ind = data.nsamples, indtest = datatest.nsamples,
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
