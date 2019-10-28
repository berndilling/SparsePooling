using Pkg; Pkg.activate("./../../SparsePooling/"); Pkg.instantiate()
push!(LOAD_PATH, "./../../SparsePooling/src/")
using SparsePooling

using Flux, BSON, PyPlot, JLD2, FileIO, LinearAlgebra, ProgressMeter, Statistics

function gettrainednet(; nettype = "CNN", dataset = "floatingMNIST")
    BSON.load(string("Reference_", nettype, "_", dataset,".bson"))[:referencenetwork]
end
function gettestdata(; targetsize = 40)
    imgs, labels, imgstest, labelstest, n_trainsamples, n_testsamples =
		getMNIST()
    reshape(zeropad(imgstest; targetsize = targetsize), targetsize^2, n_testsamples)
end

function _shiftimage(img, amplitude, nettype)
	img_s = Int(sqrt(size(img, 1)))
    if nettype == "CNN"
	    reshape(circshift(reshape(img,img_s,img_s), amplitude), img_s, img_s, 1, 1)
    else
        reshape(circshift(reshape(img,img_s,img_s), amplitude), img_s ^ 2, 1)
    end
end

function getcorrelatesperimage(m, l, img, nettype; margin = 9)
    rep = deepcopy(m[1:l](_shiftimage(img, [0, 0], nettype)).data)
    correlates = []
    for x in -margin:margin
        #for y in -margin:margin
        y = 0
            rep_shifted = deepcopy(m[1:l](_shiftimage(img, [x, y], nettype)).data)
            push!(correlates, dot(rep, rep_shifted) / dot(rep, rep))
        #end
    end
    return correlates
end

function getcorrelates(m, l, imgs, nettype; margin = 9, nofsamples = 1000)#size(imgs)[end])
    correlates = zeros(2 * margin + 1, nofsamples)
    print(string("\n calculate correlates for layer ", l, "\n"))
    @showprogress for i in 1:nofsamples
        correlates[:, i] = getcorrelatesperimage(m, l, imgs[:, i], nettype; margin = margin)
    end
    return correlates
end

function averageoversamples(correlates)
    avrg_correlates = zeros(size(correlates, 1))
    for i in 1:size(correlates, 1)
        avrg_correlates[i] = mean(correlates[i, :])
    end
    return avrg_correlates
end

function getaveragedcorrelates(m, layer, imgs, nettype; margin = 9)
    correlates = getcorrelates(m, layer, imgs, nettype)
    averageoversamples(correlates)
end

function main(nettype; margin = 9)
    m = gettrainednet(; nettype = nettype)
    testimgs = gettestdata()
    figure()
    xlabel("shift (pixels)")
    ylabel("correlation")
    title(string("Correlates for network type ", nettype))
    lowestlayer = (nettype == "CNN") ? 1 : 2
    for layer in length(m):-1:lowestlayer
        avrg_correlates = getaveragedcorrelates(m, layer, testimgs, nettype; margin = 9)
        plot(collect(-margin:margin), avrg_correlates, label = string("layer ", (nettype == "CNN") ? layer : layer - 1))
    end
    legend()
    savefig(string("invariancetest_", nettype,".pdf"))
end

#main("SP")
#main("RP")
#main("MLP")
main("CNN")
