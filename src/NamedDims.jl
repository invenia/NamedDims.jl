module NamedDims
using Base: @propagate_inbounds
using Base.Broadcast:
    Broadcasted, BroadcastStyle, DefaultArrayStyle, AbstractArrayStyle, Unknown
using LinearAlgebra
using Pkg
using Requires
using Statistics

export NamedDimsArray, dim, rename, unname, dimnames

function __init__()
    # NOTE: NamedDims is only compatible with Tracker v0.2.2; but no nice way to enforce that.
    @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("tracker_compat.jl")
end

# We use CoVector to workout if we are taking the tranpose of a tranpose etc
const CoVector = Union{Adjoint{<:Any, <:AbstractVector}, Transpose{<:Any, <:AbstractVector}}

include("name_core.jl")
include("wrapper_array.jl")
include("broadcasting.jl")
include("functions.jl")
include("functions_dims.jl")
include("functions_math.jl")

@deprecate names dimnames false

end # module
