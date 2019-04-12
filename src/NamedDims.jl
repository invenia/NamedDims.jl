module NamedDims
using Base: @propagate_inbounds
using Statistics

export NamedDimsArray, name2dim, dim_names

include("name_core.jl")
include("wrapper_array.jl")
include("functions.jl")

end # module
