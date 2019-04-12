module NamedDims
using Statistics

export NamedDimsArray, name2dim, dim_names

include("name_core.jl")
include("wrapper_array.jl")
include("functions.jl")

end # module
