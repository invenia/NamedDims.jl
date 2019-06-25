module NamedDims
using Base: @propagate_inbounds
using Base.Broadcast:
    Broadcasted, BroadcastStyle, DefaultArrayStyle, AbstractArrayStyle, Unknown
using LinearAlgebra
using Statistics

export NamedDimsArray, dim, rename, unname

# We use CoVector to workout if we are taking the tranpose of a tranpose etc
const CoVector = Union{Adjoint{<:Any, <:AbstractVector}, Transpose{<:Any, <:AbstractVector}}

include("name_core.jl")
include("wrapper_array.jl")
include("broadcasting.jl")
include("functions.jl")
include("functions_dims.jl")
include("functions_math.jl")

include("compat.jl")
end # module
