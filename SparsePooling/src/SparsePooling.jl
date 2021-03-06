
module SparsePooling
using Revise, ProgressMeter, Distributions, LinearAlgebra, Statistics, JLD2, FileIO, HDF5, MAT

#path to submodules
path = "./sparsepooling/"

include(string(path,"nonlins.jl"))
include(string(path,"types.jl"))
include(string(path,"helpers.jl"))
include(string(path,"dataimport.jl"))
include(string(path,"forwardprop.jl"))
include(string(path,"parameterupdate.jl"))
include(string(path,"learning.jl"))
include(string(path,"bar_input_generator.jl"))
include(string(path,"box_hierarchy_input_generator.jl"))

end
