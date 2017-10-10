
#####################################################
#Weight update functions

function _normalize_inputweights!(weights)
	for j in 1:size(weights)[1]
		weights[j,:] *= 1./norm(weights[j,:])
	end
end

#Algorithm for sparse coding as proposed in PhD-Thesis of C. Brito Ch2, Methods
#Implements the second part of algorithm: weight update
#omitnonlocal: boolian if non local term should be omitted:
#true: will be omitted
#false: will be taken into account
function backprop_sparse_Brito!(net::AE_sparse, learningrate; omitnonlocal = true)
	if net.nl > 1
		error("Network/AE is deeper than expected for a sparse network (more than one hidden layer)!\n")
	end
	if omitnonlocal
		for i in 1:net.nl
			#BLAS.ger!(learningrate[i],net.ax[i],net.x[i],net.w[i])
			#Above: might be useful for ordinary sparse coding à la mairal which uses .ax and .x differently!
			BLAS.ger!(learningrate[i],net.x[i+1],net.x[i],net.w[i])
			#BLAS.ger!(learningrate[i],net.x[i+1].*(net.x[i+1]-0.1),net.x[i],net.w[i]) #quadratic Hebbian
			_normalize_inputweights!(net.w[i])
	  end
	else
		for i in 1:net.nl
			#BLAS.ger!(learningrate[i],net.ax[i],(net.x[i]-BLAS.gemv('T',net.w[i],net.ax[i])),net.w[i])
			BLAS.ger!(learningrate[i],net.x[i+1],(net.x[i]-BLAS.gemv('T',net.w[i],net.x[i+1])),net.w[i])
			_normalize_inputweights!(net.w[i])
		end
	end
end
