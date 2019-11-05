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
    if nettype[1:2] == "CN"
	    reshape(circshift(reshape(img,img_s,img_s), amplitude), img_s, img_s, 1, 1)
    else
        reshape(circshift(reshape(img,img_s,img_s), amplitude), img_s ^ 2, 1)
    end
end

function getcorrelatesperimage(m, l, img, nettype;
                                margin = 9, shifts = -margin:margin,
                                singleneurons = (false, 1))
    refshift = [rand(shifts), 0]
    refrep = deepcopy(m[1:l](_shiftimage(img, refshift, nettype)).data)
    correlates = ones(length(shifts)) .* Inf
    for x in shifts
        relativeshift = x - refshift[1]
        #for y in -margin:margin
        y = 0
        if relativeshift in shifts
            index = relativeshift - collect(shifts)[1] + 1
            rep_shifted = deepcopy(m[1:l](_shiftimage(img, [x, y], nettype)).data)
            if singleneurons[1]
                inds = singleneurons[2]
                refrep = refrep[inds]
                rep_shifted = rep_shifted[inds]
            end
            dotprod = dot(refrep, rep_shifted)
            normprod = norm(refrep) * norm(rep_shifted)
            if dotprod == 0. && normprod == 0.
                cor = 0.
            else
                cor = dotprod / normprod
            end
            correlates[index] = cor
        end
        #end
    end
    return correlates
end

function getcorrelates(m, l, imgs, nettype; margin = 9, nofsamples = 1000, # size(imgs)[end])
                        singleneurons = (false, 1))
    correlates = zeros(2 * margin + 1, nofsamples)
    print(string("\n calculate correlates for layer ", l, "\n"))
    @showprogress for i in 1:nofsamples
        correlates[:, i] = getcorrelatesperimage(m, l, imgs[:, i], nettype;
                                                margin = margin, singleneurons = singleneurons)
    end
    return correlates
end

function averageoversamples(correlates)
    avrg_correlates = zeros(size(correlates, 1))
    for i in 1:size(correlates, 1)
        inds_in_range = findall(correlates[i, :] .!= Inf)
        avrg_correlates[i] = mean(correlates[i, inds_in_range])
    end
    return avrg_correlates
end

function getaveragedcorrelates(m, layer, imgs, nettype; margin = 9, singleneurons = (false, 1))
    correlates = getcorrelates(m, layer, imgs, nettype; margin = margin, singleneurons = singleneurons)
    averageoversamples(correlates)
end

function main(nettype; margin = 9, singleneurons = (false, 1))
    m = gettrainednet(; nettype = nettype)
    testimgs = gettestdata()
    figure()
    xlabel("shift (pixels)")
    ylabel("correlation")
    title(string("Correlates for network type ", nettype))
    lowestlayer = (nettype[1:2] == "CN") ? 1 : 2
    for layer in length(m):-1:lowestlayer
        avrg_correlates = getaveragedcorrelates(m, layer, testimgs, nettype; margin = margin, singleneurons = singleneurons)
        plot(collect(-margin:margin), avrg_correlates, label = string((nettype[1:2] == "CN") ? layer : layer - 1," ",string(m[layer])[1:minimum([end,50])]))
    end
    legend(fontsize = 5, loc="upper left", bbox_to_anchor=(0., -0.2))
    tight_layout()

    #savefig(string("invariancetest_", nettype,".pdf"))
end

# main("SP")
# main("RP")
# main("MLP")
main("CNN")
#main("CNN_convpool")
#main("CNN_nopool")
