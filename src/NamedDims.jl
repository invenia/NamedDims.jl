module NamedDims
using Base: @propagate_inbounds
using LinearAlgebra
using Statistics

export NamedDimsArray, dim, rename

include("name_core.jl")
include("wrapper_array.jl")
include("broadcasting.jl")
include("functions.jl")
include("functions_dims.jl")
include("functions_math.jl")

include("compat.jl")
end # module
