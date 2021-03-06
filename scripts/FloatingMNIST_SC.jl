using Pkg; Pkg.activate("./../SparsePooling/"); Pkg.instantiate()
push!(LOAD_PATH, "./../SparsePooling/src/")
using SparsePooling
using Dates, Serialization
using Pkg; Pkg.add(PackageSpec(name = "BlackBoxOptim", rev = "master"))
using BlackBoxOptim

# layer wise training + classifier training + testing
# returns scalar fitness (i.e. test error)
function trainandtest(data, datatest, ind, ind_t, layertypes;
                            nfilters1 = 5,
                            ksize1 = 5,
                            str1 = 2,
                            tau2 = 5.,
                            p1 = 0.1,
                            shiftdata = true)
    network = net(layertypes,
                [getindim(data),Int(nfilters1)],
                [0,Int(ksize1)], # kernel sizes
                [0,Int(str1)], #s strides: stride 1 in first layer works best so far
                [100.,100.], # time scales tau for SFA (ignored for non SFA layers)
   			    [0,p1], # ps: sparsity parameters p
                weight_sharing = true)

    inputfunction = getsmallimg
    intermediatestates = []
    learn_net_layerwise!(network, data, intermediatestates,
        [10^5],
        [inputfunction for i in 1:network.nr_layers],
        # TODO use getstaticimagefloatingMNIST
        [getstaticimagefloatingMNIST];
        LearningFromLayer = 2,
        LearningUntilLayer = network.nr_layers)

    if shiftdata
        ShiftPaddedMNIST!(data) # create (fixed through fixed seed) shifted data set
        ShiftPaddedMNIST!(datatest)
    end

    lasthiddenrepstrain = labelleddata(generatehiddenreps!(network, data;
                                            ind = ind, normalize = true,
                                            subtractmean = false),
                                        data.labels[1:ind]; classes = data.classes)
    lasthiddenrepstest = labelleddata(generatehiddenreps!(network, datatest;
                                            ind = ind_t, normalize = true,
                                            subtractmean = false),
                                        datatest.labels[1:ind_t]; classes = datatest.classes)
    error_train, error_test = traintopendclassifier!(network, lasthiddenrepstrain, lasthiddenrepstest; hidden_sizes = Int64[],
                iters = 10^5, ind = ind, indtest = ind_t, n_classes = length(data.classes))
    return error_train, error_test, network, data
end
function SparsePoolingSim(layertypes; nfilters1 = 10,
                            ksize1 = 8,
                            str1 = 2,
                            p1 = 0.4,
                            shiftdata = true)
    # load data
    data, datatest, ind, ind_t = getPaddedMNIST() # getNORB()
    # train model
    error_train, error_test, network, data = trainandtest(data, datatest, ind, ind_t,
                                layertypes;
                                nfilters1 = nfilters1,
                                ksize1 = ksize1,
                                str1 = str1,
                                p1 = p1,
                                shiftdata = shiftdata)
    return error_train, error_test, network, data
end

##

error_train, error_test, network, data = SparsePoolingSim(["input","sparse_patchy"];
                                                            nfilters1 = 32,
                                                            ksize1 = 5,
                                                            str1 = 1,
                                                            p1 = 0.1,
                                                            shiftdata = true);
