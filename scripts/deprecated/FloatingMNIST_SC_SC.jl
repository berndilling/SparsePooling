using Pkg; Pkg.activate("./../SparsePooling/");# Pkg.instantiate()
push!(LOAD_PATH, "./../SparsePooling/src/")
using SparsePooling
using Dates, Serialization
using Pkg; Pkg.add(PackageSpec(name = "BlackBoxOptim", rev = "master"))
using BlackBoxOptim

# layer wise training + classifier training + testing
# returns scalar fitness (i.e. test error)
function trainandtest(data, datatest, ind, ind_t;
                            nfilters1 = 5, nfilters2 = 5,
                            ksize1 = 5, ksize2 = 2,
                            str1 = 2, str2 = 1,
                            p1 = 0.1, p2 = 0.5)
    network = net(["input","sparse_patchy","sparse_patchy"],
                [getindim(data),Int(nfilters1), Int(nfilters2)],
                [0,Int(ksize1),Int(ksize2)], # kernel sizes
                [0,Int(str1),Int(str2)], #s strides: stride 1 in first layer works best so far
                [100.,100.,100.], # time scales tau for SFA (ignored for non SFA layers)
   			    [0,p1,p2], # ps: sparsity parameters p
                )

    inputfunction = getsmallimg
    intermediatestates = []
    learn_net_layerwise!(network, data, intermediatestates,
        [3*10^4, 3*10^4],
        [inputfunction for i in 1:network.nr_layers],
        # TODO use getstaticimagefloatingMNIST
        [getstaticimagefloatingMNIST, getstaticimagefloatingMNIST];
        LearningFromLayer = 2,
        LearningUntilLayer = network.nr_layers)

    ShiftPaddedMNIST!(data) # create (fixed through fixed seed) shifted data set
    ShiftPaddedMNIST!(datatest)
    # TODO does it make sense to train the classifier on this fixed data set?
    lasthiddenrepstrain = labelleddata(generatehiddenreps!(network, data;
                                            ind = ind, normalize = true,
                                            subtractmean = false),
                                        data.labels[1:ind]; classes = data.classes)
    lasthiddenrepstest = labelleddata(generatehiddenreps!(network, datatest;
                                            ind = ind_t, normalize = true,
                                            subtractmean = false),
                                        datatest.labels[1:ind_t]; classes = datatest.classes)
    error_train, error_test = traintopendclassifier!(network, lasthiddenrepstrain, lasthiddenrepstest; hidden_sizes = Int64[],
                iters = 10^6, ind = ind, indtest = ind_t, n_classes = length(data.classes))
    return error_train, error_test, network, data
end
function SparsePoolingSim(; nfilters1 = 10, nfilters2 = 20,
                            ksize1 = 8, ksize2 = 2,
                            str1 = 2, str2 = 2,
                            p1 = 0.2, p2 = 0.2)
    # load data
    data, datatest, ind, ind_t = getPaddedMNIST() # getNORB()
    # train model
    error_train, error_test, network, data = trainandtest(data, datatest, 10000, 10000; # ind, ind_t;
                                nfilters1 = nfilters1, nfilters2 = nfilters2,
                                ksize1 = ksize1, ksize2 = ksize2,
                                str1 = str1, str2 = str2,
                                p1 = p1, p2 = p2)
    return error_train, error_test, network, data
end

##

error_train, error_test, network, data = SparsePoolingSim();
