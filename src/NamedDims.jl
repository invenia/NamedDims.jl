module NamedDims
using Base: @propagate_inbounds
using Base.Broadcast:
    Broadcasted, BroadcastStyle, DefaultArrayStyle, AbstractArrayStyle, Unknown
using ChainRulesCore
using CovarianceEstimation
using LinearAlgebra
using AbstractFFTs
using Pkg
using Requires
using Statistics

export NamedDimsArray, dim, rename, unname, dimnames

function __init__()
    # NOTE: NamedDims is only compatible with Tracker v0.2.2; but no nice way to enforce that.
    @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("tracker_compat.jl")
end

# We use CoVector to workout if we are taking the tranpose of a tranpose etc
const CoVector = Union{Adjoint{<:Any,<:AbstractVector},Transpose{<:Any,<:AbstractVector}}

include("name_core.jl")
include("wrapper_array.jl")
include("show.jl")
include("name_operations.jl")

include("broadcasting.jl")
include("chainrules.jl")

include("functions.jl")
include("functions_dims.jl")
include("functions_math.jl")
include("cat.jl")
include("fft.jl")

@deprecate names dimnames false
@deprecate refine_names NamedDimsArray true

include("functions_linearalgebra.jl")

end # module
