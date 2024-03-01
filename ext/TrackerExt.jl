module TrackerExt

isdefined(Base, :get_extension) ? (using Tracker) : (using ..Tracker)
using NamedDims: NamedDims, dimnames, NamedDimsStyle, NamedDimsArray, @declare_matmul

# The following blocks ever constructing TrackedArrays of NamedDimArrays.
# This is not strictly forbidden (thus is commented out) but is useful for debugging things
#==
Tracker.TrackedArray(::NamedDimsArray) = error("Should not make Tracked NamedDimsArray")
Tracker.TrackedArray(::Tracker.Call, ::NamedDimsArray) = error("Should not make Tracked NamedDimsArray")
Tracker.TrackedArray(::Tracker.Call, ::NamedDimsArray, ::AbstractArray) = error("Should not make Tracked NamedDimsArray")
Tracker.TrackedArray(::Tracker.Call, ::AbstractArray, ::NamedDimsArray) = error("Should not make Tracked NamedDimsArray")
==#

function Base.BroadcastStyle(::NamedDimsStyle{A}, b::Tracker.TrackedStyle) where {A}
    return NamedDimsStyle(A(), b)
end
function Base.BroadcastStyle(a::Tracker.TrackedStyle, ::NamedDimsStyle{B}) where {B}
    return NamedDimsStyle(a, B())
end

@declare_matmul(Tracker.TrackedMatrix, Tracker.TrackedVector)

function Tracker.data(nda::NamedDimsArray{L}) where {L}
    content = Tracker.data(parent(nda))
    return NamedDimsArray{L}(content)
end

function Tracker.track(c::Tracker.Call, nda::NamedDimsArray{L}) where {L}
    content = Tracker.track(c, parent(nda))
    return NamedDimsArray{L}(content)
end

for f in (:forward, :back, :back!, :grad, :istracked, :tracker)
    @eval function Tracker.$f(nda::NamedDimsArray, args...)
        return Tracker.$f(parent(nda), args...)
    end
end

end
