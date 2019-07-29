using Pkg; Pkg.activate("./../../SparsePooling/"); Pkg.instantiate()
push!(LOAD_PATH, "./../../SparsePooling/src/")
using SparsePooling
using Dates, Serialization

#####
##### helpers
#####

struct FitParameters
    values::Vector{Float64}
    symbols::Vector{Symbol}
    mapping::Dict{Symbol, Int}
end
FitParameters(symbols) = FitParameters(zeros(length(symbols)),
                                       symbols,
                                       Dict(Pair(s, i)
                                            for (s, i) in zip(symbols, 1:length(symbols))))
function setvalues!(fp::FitParameters, values)
    fp.values .= values
    fp
end
Base.setindex!(fp::FitParameters, value, key) = fp.values[fp.mapping[key]] = value
Base.getindex(fp::FitParameters, key) = fp.values[fp.mapping[key]]
@inline function Base.iterate(fp::FitParameters, i = 1)
    length(fp.values) == i && return nothing
    Pair(fp.symbols[i], fp.values[i]), i + 1
end
function Base.show(io::IO, mime::MIME"text/plain", fp::FitParameters)
    println("$(length(fp.values))-element FitParameters")
    width = maximum((x -> length(string(x))).(fp.symbols))
    for (s, v) in zip(fp.symbols, fp.values)
        println(" $(lpad(s, width)) = $v")
    end
end
function optimization_wrapper(func, fp;
                              log_name = tempname(),
                              log_timeinterval = 20,
                              log_stepinterval = Inf)
    t0 = time()
    i = 0
    optval = Inf
    optimizer = fp
    logfilename = log_name * ".log"
    paramfilename = log_name * ".fp"
    @info "Logging to $logfilename."
    write(logfilename, "Optimizing parameters $(optimizer.symbols).\n")
    x -> begin
        i += 1
        t1 = time()
        if t1 - t0 > log_timeinterval || i % log_stepinterval == 0
            t0 = t1
            open(logfilename, "a") do logfile
                println(logfile, "$(rpad(now(), 23))  $(lpad(i, 5))  $(rpad(optval, 18)) $(optimizer.values)")
            end
            open(paramfilename, "w") do paramfile
                serialize(paramfile, optimizer)
            end
        end
        setvalues!(fp, x)
        res = func(; fp...)
        if res < optval
            optval = res
            optimizer = deepcopy(fp)
        end
        res
    end
end


####
#### SparsePooling optimisation function definition
####

using Pkg; Pkg.add(PackageSpec(name = "BlackBoxOptim", rev = "master"))
using BlackBoxOptim

# structs ... 

train() = ... # layer wise training + classifier training
test() = ... # should return scalar fitness (e.g. test error)
function SparsePoolingSim()
    # ...
    test()
    train()
end

###############################################################################
## Run optim
###############################################################################

fp = FitParameters([...])
setup = bbsetup(optimization_wrapper(SparsePoolingSim, fp,
                                     log_stepinterval = 10,
                                     log_name = "test"),
                SearchSpace = RectSearchSpace([(4., 5.), (0., 4.),
                                               (2., 8.), (2.1, 7.), (-.1, 4.)],
                                              dimdigits = [0, -1,
                                                           1, -1, -1]),
                MaxSteps = 10^3, TraceMode = :Silent);
# dimdigits: 0 = only integer values are allowed (for discrete options),
#           -1 = machine precision
@time res = bboptimize(setup);
setvalues!(fp, best_candidate(res))
fp2 = deserialize("test.fp")
readlines("test.log")
