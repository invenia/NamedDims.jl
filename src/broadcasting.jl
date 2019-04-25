# See: https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting-1

"""
    NamedDimsStyle{S}
This is a `BroadcastStyle` for NamedDimsArray's
It preserves the dimension names.
`S` should be the `BroadcastStyle` of the wrapped type.
"""
struct NamedDimsStyle{S <: Base.BroadcastStyle} <: Base.Broadcast.AbstractArrayStyle{Any} end
NamedDimsStyle(::S) where {S} = NamedDimsStyle{S}()
NamedDimsStyle(::S, ::Val{N}) where {S,N} = NamedDimsStyle(S(Val(N)))
NamedDimsStyle(::Val{N}) where N = NamedDimsStyle{Broadcast.DefaultArrayStyle{N}}()
function NamedDimsStyle(a::Base.BroadcastStyle, b::Base.BroadcastStyle)
    inner_style = Base.BroadcastStyle(a, b)

    # if the inner_style is Unknow then so is the outer-style
    if inner_style isa Broadcast.Unknown
        return Broadcast.Unknown()
    else
        return NamedDimsStyle(inner_style)
    end
end
function Base.BroadcastStyle(::Type{<:NamedDimsArray{L, T, N, A}}) where {L, T, N, A}
    inner_style = typeof(Base.BroadcastStyle(A))
    return NamedDimsStyle{inner_style}()
end

function Base.BroadcastStyle(::NamedDimsStyle{A}, ::NamedDimsStyle{B}) where {A, B}
    return NamedDimsStyle(A(), B())
end
function Base.BroadcastStyle(::NamedDimsStyle{A}, b::B) where {A, B <: Broadcast.BroadcastStyle}
    return NamedDimsStyle(A(), b)
end
function Base.BroadcastStyle(a::A, ::NamedDimsStyle{B}) where {A <: Broadcast.BroadcastStyle, B}
    return NamedDimsStyle(a, B())
end



function Broadcast.broadcasted(::NamedDimsStyle{S}, f, args...) where S
    # Delgate to inner style
    inner = Broadcast.broadcasted(S(), f, args...)
    if inner isa Broadcast.Broadcasted
        return Broadcast.Broadcasted{NamedDimsStyle{S}}(inner.f, inner.args, inner.axes)
    else # eagerly evaluated
        return inner
    end
end


function Base.similar(
    bc::Broadcast.Broadcasted{NamedDimsStyle{S}},
    ::Type{T}
) where {S,T}
    inner_bc = Broadcast.Broadcasted{S}(bc.f, bc.args, bc.axes)
    data = similar(inner_bc, T)

    L = broadcasted_names(bc)
    return NamedDimsArray{L}(data)
end


broadcasted_names(bc::Broadcast.Broadcasted) = broadcasted_names(bc.args...)
function broadcasted_names(a, bs...)
    a_name = broadcasted_names(a)
    b_name = broadcasted_names(bs...)
    unify_names_longest(a_name, b_name)
end
broadcasted_names(a::AbstractArray) = names(a)
broadcasted_names(a) = tuple()



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


function Base.similar(
    bc::Broadcast.Broadcasted{TrackedStyle},
    ::Type{TrackedReal{T}}
) where {S,T}
    # This is not great, as it loose what type Tracker was wrapping
    # but TrackedStyle doesn't really keep track of that
    data = similar(Array{T}, axes(bc))
    return TrackedArray(data)
end

