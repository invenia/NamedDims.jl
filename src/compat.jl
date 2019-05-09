# Tracker.jl Compat
using Tracker
using Tracker: TrackedStyle, TrackedReal

# The following blocks ever constructing TrackedArrays of NamedDimArrays.
# This is not strictly forbidden (thus is commented out) but is useful for debugging things
#==
Tracker.TrackedArray(::NamedDimsArray) = error("Should not make Tracked NamedDimsArray")
Tracker.TrackedArray(::Tracker.Call, ::NamedDimsArray) = error("Should not make Tracked NamedDimsArray")
Tracker.TrackedArray(::Tracker.Call, ::NamedDimsArray, ::AbstractArray) = error("Should not make Tracked NamedDimsArray")
Tracker.TrackedArray(::Tracker.Call, ::AbstractArray, ::NamedDimsArray) = error("Should not make Tracked NamedDimsArray")
==#

Base.BroadcastStyle(::NamedDimsStyle{A}, b::TrackedStyle) where {A} = NamedDimsStyle(A(), b)
Base.BroadcastStyle(a::TrackedStyle, ::NamedDimsStyle{B}) where {B} = NamedDimsStyle(a, B())

function Base.:*(
    a::Tracker.TrackedArray{T, 2},
    b::NamedDims.NamedDimsArray{L, S, 2}
) where {T, L, S}
    return NamedDimsArray{NamedDims.names(a)}(a) * b
end

function Base.:*(
    a::NamedDims.NamedDimsArray{L, S, 2},
    b::Tracker.TrackedArray{T, 2}
) where {T, L, S}
    return a * NamedDimsArray{NamedDims.names(b)}(b)
end


function Tracker.data(nda::NamedDimsArray{L}) where L
    content = Tracker.data(parent(nda))
    return NamedDimsArray{L}(content)
end

function Tracker.track(c::Tracker.Call, nda::NamedDimsArray{L}) where L
    content = Tracker.track(c, parent(nda))
    return NamedDimsArray{L}(content)
end

for f in (:forward, :back, :back!, :grad, :istracked, :tracker)
    @eval function Tracker.$f(nda::NamedDimsArray, args...)
        return Tracker.$f(parent(nda), args...)
    end
end
