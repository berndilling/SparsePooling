using Pkg; Pkg.activate("./../SparsePooling/"); Pkg.instantiate()
push!(LOAD_PATH, "./../SparsePooling/src/")
using SparsePooling
using Dates, Serialization
using Pkg; Pkg.add(PackageSpec(name = "BlackBoxOptim", rev = "master"))
using BlackBoxOptim
using JLD2, FileIO

# layer wise training + classifier training + testing
# returns scalar fitness (i.e. test error)
function trainandtest(data, datatest, ind, ind_t, layertypes;
                            nfilters = [5, 5],
                            ksize = [5, 2],
                            str = [1, 1],
                            tau = [100., 5.],
                            p = [0.1, 0.5],
                            shiftdata = true)
    network = net(layertypes,
                vcat(getindim(data), [Int(nfilters[i]) for i in 1:length(nfilters)]),
                vcat(0, [Int(ksize[i]) for i in 1:length(ksize)]), # kernel sizes
                vcat(0, [Int(str[i]) for i in 1:length(str)]), # strides: stride 1 in first layer works best so far
                vcat(100., [tau[i] for i in 1:length(tau)]), # time scales tau for SFA (ignored for non SFA layers)
   			    vcat(0, [p[i] for i in 1:length(p)]); # ps: sparsity parameters p
                weight_sharing = true)

    inputfunction = getsmallimg
    intermediatestates = []
    learn_net_layerwise!(network, data, intermediatestates,
        #[10^4, 1, 10^4, 1, 10^4, 1],
        [10^4, 1, 10^4, 1, 10^4, 1, 10^4, 1],
        [inputfunction for i in 1:network.nr_layers],
        [getstaticimagefloatingMNIST for i in 1:network.nr_layers]; # getmovingimage
        LearningFromLayer = 2,
        LearningUntilLayer = network.nr_layers,
        dynamic = shiftdata)

    if shiftdata
        ShiftPaddedMNIST!(data) # create (fixed through fixed seed) shifted data set
        ShiftPaddedMNIST!(datatest)
    end    # TODO does it make sense to train the classifier on this fixed data set?

    lasthiddenrepstrain = labelleddata(generatehiddenreps!(network, data;
                                            ind = ind, normalize = true,
                                            subtractmean = false),
                                        data.labels[1:ind]; classes = data.classes)
    lasthiddenrepstest = labelleddata(generatehiddenreps!(network, datatest;
                                            ind = ind_t, normalize = true,
                                            subtractmean = false),
                                        datatest.labels[1:ind_t]; classes = datatest.classes)
    error_train, error_test = traintopendclassifier!(network, lasthiddenrepstrain, lasthiddenrepstest; hidden_sizes = Int64[],
                iters = 10^7, # 10^7
                ind = ind, indtest = ind_t, n_classes = length(data.classes))
    return error_train, error_test, network, data, lasthiddenrepstrain, lasthiddenrepstest
end
function SparsePoolingSim(layertypes; nfilters = [10, 10],
                            ksize = [3, 2],
                            str = [1, 2],
                            tau = [100., 5.],
                            p = [0.1, 0.5],
                            shiftdata = true)

    data, datatest, ind, ind_t = getPaddedMNIST() # # getNORB() getMNIST()# getPaddedMNIST() #getCIFAR10(; greyscale = false)#

    error_train, error_test, network, data, hrtrain, hrtest = trainandtest(data, datatest, ind, ind_t, #
                                layertypes;
                                nfilters = nfilters,
                                ksize = ksize,
                                str = str,
                                tau = tau,
                                p = p,
                                shiftdata = shiftdata)
    return error_train, error_test, network, data, hrtrain, hrtest
end

# error_train, error_test, network, data, hrtrain, hrtest = SparsePoolingSim(vcat("input", "sparse_patchy", "max_pool_patchy", "sparse_patchy", "max_pool_patchy", "sparse_patchy", "max_pool_patchy"); #"input_color"
#                                                             nfilters = [32, 32, 64, 64, 128, 128],
#                                                             ksize = [3, 2, 3, 2, 3, 2],
#                                                             str = [1, 2, 1, 2, 1, 2], #[2, 1, 2, 1, 2, 1], # downsampling in convlayers! Otherwise use [1, 2, 1, 2, 1, 2] (89/88 % acc on floatingMNIST)!
#                                                             tau = [100, 5., 100., 5., 100., 5.],
#                                                             p = [0.1, 0.5, 0.1, 0.5, 0.2, 0.5], # 0.1, 0.1, 0.2 for SC layers lead to 90.6/90%
#                                                             shiftdata = true)

error_train, error_test, network, data, hrtrain, hrtest = SparsePoolingSim(vcat("input", "sparse_patchy", "max_pool_patchy", "sparse_patchy", "max_pool_patchy", "sparse_patchy", "max_pool_patchy", "sparse_patchy", "max_pool_patchy"); #"input_color"
                                                            nfilters = [32, 32, 64, 64, 128, 128, 256, 256],
                                                            ksize = [3, 2, 3, 2, 3, 2, 2, 2],
                                                            str = [1, 2, 1, 2, 1, 2, 1, 1],
                                                            tau = [100, 5., 100., 5., 100., 5., 100., 5.],
                                                            p = [0.1, 0.5, 0.1, 0.5, 0.2, 0.5, 0.2, 0.5],
                                                            shiftdata = true)

save("./floatingMNIST/FloatingMNIST_stack_SC_maxpool_4_modules.jld2", "error_train", error_train, "error_test", error_test, "network", network, "data", data, "hrtrain", hrtrain, "hrtest", hrtest)
