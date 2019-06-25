# Tracker.jl Compat

# The following blocks ever constructing TrackedArrays of NamedDimArrays.
# This is not strictly forbidden (thus is commented out) but is useful for debugging things
#==
Tracker.TrackedArray(::NamedDimsArray) = error("Should not make Tracked NamedDimsArray")
Tracker.TrackedArray(::Tracker.Call, ::NamedDimsArray) = error("Should not make Tracked NamedDimsArray")
Tracker.TrackedArray(::Tracker.Call, ::NamedDimsArray, ::AbstractArray) = error("Should not make Tracked NamedDimsArray")
Tracker.TrackedArray(::Tracker.Call, ::AbstractArray, ::NamedDimsArray) = error("Should not make Tracked NamedDimsArray")
==#

Base.BroadcastStyle(::NamedDimsStyle{A}, b::Tracker.TrackedStyle) where {A} = NamedDimsStyle(A(), b)
Base.BroadcastStyle(a::Tracker.TrackedStyle, ::NamedDimsStyle{B}) where {B} = NamedDimsStyle(a, B())

@declare_matmul(Tracker.TrackedMatrix, Tracker.TrackedVector)

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
