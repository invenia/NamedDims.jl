module NamedDims
using Base: @propagate_inbounds
using Base.Broadcast:
    Broadcasted, BroadcastStyle, DefaultArrayStyle, AbstractArrayStyle, Unknown
using LinearAlgebra
using Pkg
using Statistics

export NamedDimsArray, dim, rename, unname, dimnames

# We use CoVector to workout if we are taking the tranpose of a tranpose etc
const CoVector = Union{Adjoint{<:Any,<:AbstractVector},Transpose{<:Any,<:AbstractVector}}

include("name_core.jl")
include("wrapper_array.jl")
include("show.jl")
include("name_operations.jl")

include("broadcasting.jl")

include("functions.jl")
include("functions_dims.jl")
include("functions_math.jl")
include("cat.jl")
include("fft.jl")

@deprecate names dimnames false
@deprecate refine_names NamedDimsArray true

include("functions_linearalgebra.jl")

@static if !isdefined(Base, :get_extension)
    using Requires
end
@static if !isdefined(Base, :get_extension)
    include("../ext/AbstractFFTsExt.jl")
    include("../ext/ChainRulesCoreExt.jl")
    include("../ext/CovarianceEstimationExt.jl")
    
    function __init__()
        # NOTE: NamedDims is only compatible with Tracker v0.2.2; but no nice way to enforce that.
        @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("../ext/TrackerExt.jl")
    end
end

end # module
