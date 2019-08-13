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

## Data import

# NORB

function getNORB()
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
    ind_test = datatest.nsamples # 5000 # 10000

    return data, datatest, ind, ind_test
end

include("./../floatingMNIST/floatingMNIST.jl")
function getPaddedMNIST(; targetsize = 50, margin = div(targetsize - 28, 2) + 3)
    smallimgs, labels, smallimgstest, labelstest, n_trainsamples, n_testsamples =
		getMNIST();
	imgs = zeropad(smallimgs; targetsize = targetsize)
	imgstest = zeropad(smallimgstest; targetsize = targetsize)

	data = labelleddata(imgs, labels, margin)
	datatest = labelleddata(imgstest, labelstest, margin)

    return data, datatest, n_trainsamples, n_testsamples
end
