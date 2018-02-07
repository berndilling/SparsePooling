#includes all relevant "modules"

#path to modules
path = "./"

#CHANGING IMPORT ORDER MIGHT CAUSE PROBLEMS!
include(string(path,"nonlins.jl"))
include(string(path,"types.jl"))
include(string(path,"helpers.jl"))
include(string(path,"dataimport.jl"))
include(string(path,"forwardprop.jl"))
include(string(path,"parameterupdate.jl"))
include(string(path,"learning.jl"))
include(string(path,"bar_input_generator.jl"))
