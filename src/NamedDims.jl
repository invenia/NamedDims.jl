module NamedDims
using Base: @propagate_inbounds
using Base.Broadcast:
    Broadcasted, BroadcastStyle, DefaultArrayStyle, AbstractArrayStyle, Unknown
using LinearAlgebra
using Pkg
using Requires
using Statistics

export NamedDimsArray, dim, rename, unname

function __init__()
    @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include_tracker_compat()
end

# For Tracker-NamedDims compatibility we require a Tracker version compatible with 0.2.2.
# Since we use Requires.jl we have no nicer way to set the Tracker compatability.
function include_tracker_compat()
    tracker_version = Pkg.installed()["Tracker"]
    allowed_versions = Pkg.Types.semver_spec("0.2.2")
    if tracker_version ∈ allowed_versions
        include("tracker_compat.jl")
    else
        @warn string(
            "Tracker version not compatible with NamedDims. ",
            "Tracker compatability functionality has been disabled."
        ) tracker_version allowed_versions
    end
end

# We use CoVector to workout if we are taking the tranpose of a tranpose etc
const CoVector = Union{Adjoint{<:Any, <:AbstractVector}, Transpose{<:Any, <:AbstractVector}}

include("name_core.jl")
include("wrapper_array.jl")
include("broadcasting.jl")
include("functions.jl")
include("functions_dims.jl")
include("functions_math.jl")

end # module
