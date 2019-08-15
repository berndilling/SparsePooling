using Pkg; Pkg.activate("./../../SparsePooling/"); Pkg.instantiate()
push!(LOAD_PATH, "./../../SparsePooling/src/")
using SparsePooling
using Dates, Serialization
using Pkg; Pkg.add(PackageSpec(name = "BlackBoxOptim", rev = "master"))
using BlackBoxOptim

# max number of steps
MaxSteps = 1000
init = ARGS[1]
log_name = "BBO_SP__SC_SFA_init$init"

###############################################################################
## Helper types and functions
###############################################################################

include("BlackBoxHelpers.jl")

###############################################################################
## Simulation function
###############################################################################

# layer wise training + classifier training + testing
# returns scalar fitness (i.e. test error)
function trainandtest(data, datatest, ind, ind_t;
                            nfilters1 = 5, nfilters2 = 5,
                            ksize1 = 5, ksize2 = 5,
                            str1 = 2, str2 = 2,
                            tau2 = 5.,
                            p1 = 0.1, p2 = 0.1)
    network = net(["input","sparse_patchy","pool_patchy"],
                [getindim(data),Int(nfilters1), Int(nfilters2)],
                [0,Int(ksize1),Int(ksize2)], # kernel sizes
                [0,Int(str1),Int(str2)], #s strides: stride 1 in first layer works best so far
                [100.,100.,tau2], # time scales tau for SFA (ignored for non SFA layers)
   			    [0,p1,p2], # ps: sparsity parameters p
                )

    inputfunction = getsmallimg
    intermediatestates = []
    learn_net_layerwise!(network, data, intermediatestates,
        [10^5, 10^4],
        [inputfunction for i in 1:network.nr_layers],
        [getstaticimage, getmovingimage];
        LearningFromLayer = 2,
        LearningUntilLayer = network.nr_layers)

    ShiftPaddedMNIST!(data)
    ShiftPaddedMNIST!(datatest)
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
    return error_test
end


function SparsePoolingSim(; nfilters1 = 5, nfilters2 = 5,
                            ksize1 = 5, ksize2 = 5,
                            str1 = 2, str2 = 2,
                            tau2 = 5, p1 = 0.1, p2 = 0.1)
    # load data
    data, datatest, ind, ind_t = getPaddedMNIST() # getNORB()
    # train model
    fitness = trainandtest(data, datatest, ind, ind_t;
                                nfilters1 = nfilters1, nfilters2 = nfilters2,
                                ksize1 = ksize1, ksize2 = ksize2,
                                str1 = str1, str2 = str2,
                                tau2 = tau2, p1 = p1, p2 = p2)
    return fitness # equals error (lower is better)
end

###############################################################################
## Actual optimisation
###############################################################################

fp = FitParameters([:nfilters1, :nfilters2,
                    :ksize1, :ksize2,
                    :str1, :str2,
                    :tau2,
                    :p1, :p2])
setup = bbsetup(optimization_wrapper(SparsePoolingSim, fp,
                                     log_stepinterval = 1,
                                     log_name = log_name),
                SearchSpace = RectSearchSpace([(4., 20.), (4., 20.),
                                               (2., 10.), (1., 3.),
                                               (1., 2.), (1., 2.),
                                               (1.,20.), # default duration of seq. is 20
                                               (1e-2, .5), (1e-2, .5)],
                                              dimdigits = [0, 0,
                                                           0, 0,
                                                           0, 0,
                                                           -1,
                                                           -1, -1]),
               MaxFuncEvals = MaxSteps);
# dimdigits: 0 = only integer values are allowed (for discrete options),
#           -1 = machine precision

# Run optimisation
@time res = bboptimize(setup);
setvalues!(fp, best_candidate(res))
fp2 = deserialize("$log_name.fp")
readlines("$log_name.log")

###############################################################################
## Single simulation (for testing)
###############################################################################

# SparsePoolingSim(; nfilters1 = 7, nfilters2 = 8,
#                             ksize1 = 10, ksize2 = 1,
#                             str1 = 1, str2 = 2,
#                             tau2 = 2, p1 = 0.3, p2 = 0.2)
