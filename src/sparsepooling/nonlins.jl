
#####################################################
#nonlinearities

@inline function relu!(inp, outp)
	for i in 1:length(inp)
		outp[i] = max(0, inp[i])
	end
end
@inline function relu_diff!(ax, error)
	for i in 1:length(ax)
		error[i] *= ax[i] > 0
	end
end

@inline function relu!(layer)
	for i in 1:length(layer.u)
		layer.a[i] = clamp(layer.u[i]-layer.t[i],0.,Inf64) #thresholded, linear rectifier
	end
end

@inline function lin!(inp, outp)
	for i in 1:length(inp)
		outp[i] = inp[i]
	end
end
@inline function lin_diff!(ax, error)
end

@inline function lin!(layer)
	for i in 1:length(layer.u)
		layer.a[i] = deepcopy(layer.u[i])
	end
end

@inline function sigm!(inp, outp)
	for i in 1:length(inp)
		outp[i] = 1./(1.+exp(-inp[i]))
	end
end
@inline function sigm_diff!(ax, error)
	for i in 1:length(ax)
		error[i] *= 1./(1+exp(-ax[i]))*(1.-1./(1+exp(-ax[i])))
	end
end

@inline function sigm!(layer; lambda = 10.)
	for i in 1:length(layer.u)
		layer.a[i] = 1./(1.+exp(-lambda * (layer.u[i]-layer.t[i])))
	end
end

@inline function heavyside!(layer)
	for i in 1:length(layer.u)
		layer.a[i] = Float64(layer.u[i] > layer.t[i])# heavyside
	end
end

@inline function pwl!(layer)
	for i in 1:length(layer.u)
		layer.a[i] = clamp(layer.u[i]-layer.t[i],0.,1.) #piece-wise linear
	end
end

@inline function ReSQRT!(layer)
	for i in 1:length(layer.u)
		layer.a[i] = sqrt(clamp(layer.u[i]-layer.t[i],0.,Inf64)) #thresholded square root
	end
end

@inline function LIFactfunct!(layer)
	# Activation function of an Leaky Integrate and Fire model (with refractatoriness)
	# IMPLIES U_RESET = 0!
	# USE ONLY IF THRESHOLDS (AND MEMBRANE POTENTIALS?) ARE ABOVE 0!
	# OneOverMaxFiringRate: Parameter for refrect.: 0 -> no refractatoriness
	# Zylberberg has 50 as maximum spike rate in his model! -> OneOverMaxFiringRate = 1/50
	for i in 1:length(layer.u)
		if layer.u[i] <= layer.t[i]
			layer.a[i] = 0.
		else
			layer.a[i] = 1./(layer.parameters.OneOverMaxFiringRate-log(1-layer.t[i]/layer.u[i]))
		end
	end
end
