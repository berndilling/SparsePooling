#includes all relevant "modules"

#path to modules
path = "./sparsepooling/"

include(string(path,"types.jl"))
include(string(path,"nonlins.jl"))
include(string(path,"helpers.jl"))
include(string(path,"dataimport.jl"))
include(string(path,"forwardprop.jl"))
include(string(path,"parameterupdate.jl"))
