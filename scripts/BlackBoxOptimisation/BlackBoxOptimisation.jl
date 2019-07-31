using Pkg; Pkg.activate("./../../SparsePooling/"); Pkg.instantiate()
push!(LOAD_PATH, "./../../SparsePooling/src/")
using SparsePooling
using Dates, Serialization
using Pkg; Pkg.add(PackageSpec(name = "BlackBoxOptim", rev = "master"))
using BlackBoxOptim

# max number of steps
MaxSteps = 100
log_name = "SP_SC_SFA"

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
                [size(data.data)[1]^2,Int(nfilters1), Int(nfilters2)],
                [0,Int(ksize1),Int(ksize2)], # kernel sizes
                [0,Int(str1),Int(str2)], #s strides: stride 1 in first layer works best so far
                [100.,100.,tau2], # time scales tau for SFA (ignored for non SFA layers)
   			    [0,p1,p2], # ps: sparsity parameters p
                )

    inputfunction = getsmallimg
    intermediatestates = []
    learn_net_layerwise!(network, data, intermediatestates,
        [10^4, 5*10^2],
        [inputfunction for i in 1:network.nr_layers],
        [getstaticimage, getmovingimage];
        LearningFromLayer = 2,
        LearningUntilLayer = network.nr_layers)

    lasthiddenrepstrain = labelleddata(generatehiddenreps!(network, data;
            ind = ind, normalize = true, subtractmean = false), data.labels[1:ind]; classes = 0:4)
    lasthiddenrepstest = labelleddata(generatehiddenreps!(network, datatest;
            ind = ind_t, normalize = true, subtractmean = false), datatest.labels[1:ind_t]; classes = 0:4)
    error_train, error_test = traintopendclassifier!(network, lasthiddenrepstrain, lasthiddenrepstest; hidden_sizes = Int64[],
                iters = 10^6, ind = ind, indtest = ind_t, n_classes = 5)
    return error_test
end


function SparsePoolingSim(; nfilters1 = 5, nfilters2 = 5,
                            ksize1 = 5, ksize2 = 5,
                            str1 = 2, str2 = 2,
                            tau2 = 5, p1 = 0.1, p2 = 0.1)
    # load data
    data, datatest, ind, ind_t = getNORB()
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
                SearchSpace = RectSearchSpace([(2., 20.), (2., 20.),
                                               (1., 10.), (1., 10.),
                                               (1., 2.), (1., 2.),
                                               (1.,10.), # default duration of seq. is 20
                                               (0., 1.), (0., 1.)],
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
