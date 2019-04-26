##################################
# Tracker.jl Compat
using Tracker
using Tracker: TrackedStyle, TrackedReal

function Base.BroadcastStyle(::NamedDimsStyle{A}, b::TrackedStyle) where {A}
    return NamedDimsStyle(A(), b)
end
function Base.BroadcastStyle(a::TrackedStyle, ::NamedDimsStyle{B}) where {B}
    return NamedDimsStyle(a, B())
end

function Base.:*(
    a::Tracker.TrackedArray{T, 2, A} where A,
    b::NamedDims.NamedDimsArray{L, S, 2, A} where A <: AbstractArray{S, 2}
) where {T, L, S}
    return NamedDimsArray{NamedDims.names(a)}(a) * b
end

function Tracker.data(nda::NamedDimsArray{L}) where{L}
    content = Tracker.data(parent(nda))
    return NamedDimsArray{L}(content)
end

#Tracker.istracked(nda::NamedDimsArray) = Tracker.istracked(parent(nda))
#Tracker.tracker(nda::NamedDimsArray) = Tracker.tracker(parent(nda))

