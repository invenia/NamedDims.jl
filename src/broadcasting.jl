"""
    NamedDimsStyle{S}
This is a `BroadcastStyle` for NamedDimsArray's
It preserves the dimension names.
`S` should be the `BroadcastStyle` of the wrapped type.
"""
struct NamedDimsStyle{S <: BroadcastStyle} <: AbstractArrayStyle{Any} end
NamedDimsStyle(::S) where {S} = NamedDimsStyle{S}()
NamedDimsStyle(::S, ::Val{N}) where {S,N} = NamedDimsStyle(S(Val(N)))
NamedDimsStyle(::Val{N}) where N = NamedDimsStyle{DefaultArrayStyle{N}}()
function NamedDimsStyle(a::BroadcastStyle, b::BroadcastStyle)
    inner_style = BroadcastStyle(a, b)

    # if the inner_style is Unknown then so is the outer-style
    if inner_style isa Unknown
        return Unknown()
    else
        return NamedDimsStyle(inner_style)
    end
end
function Base.BroadcastStyle(::Type{<:NamedDimsArray{L, T, N, A}}) where {L, T, N, A}
    inner_style = typeof(BroadcastStyle(A))
    return NamedDimsStyle{inner_style}()
end


Base.BroadcastStyle(::NamedDimsStyle{A}, ::NamedDimsStyle{B}) where {A, B} = NamedDimsStyle(A(), B())
Base.BroadcastStyle(::NamedDimsStyle{A}, b::B) where {A, B} = NamedDimsStyle(A(), b)
Base.BroadcastStyle(a::A, ::NamedDimsStyle{B}) where {A, B} = NamedDimsStyle(a, B())
Base.BroadcastStyle(::NamedDimsStyle{A}, b::DefaultArrayStyle) where {A} = NamedDimsStyle(A(), b)
Base.BroadcastStyle(a::AbstractArrayStyle{M}, ::NamedDimsStyle{B}) where {B,M} = NamedDimsStyle(a, B())


"""
    unwrap_broadcasted

Recursively unwraps `NamedDimsArray`s and `NamedDimsStyle`s.
replacing the `NamedDimsArray`s with the wrapped array,
and `NamedDimsStyle` with the wrapped `BroadcastStyle`.
"""
function unwrap_broadcasted(bc::Broadcasted{NamedDimsStyle{S}}) where S
    inner_args = map(unwrap_broadcasted, bc.args)
    return Broadcasted{S}(bc.f, inner_args)
end
unwrap_broadcasted(x) = x
unwrap_broadcasted(nda::NamedDimsArray) = parent(nda)


# We need to implement copy because if the wrapper array type does not support setindex
# then the `similar` based default method will not work
function Broadcast.copy(bc::Broadcasted{NamedDimsStyle{S}}) where S
    inner_bc = unwrap_broadcasted(bc)
    data = copy(inner_bc)

    L = broadcasted_names(bc)
    return NamedDimsArray{L}(data)
end

function Base.copyto!(dest::AbstractArray, bc::Broadcasted{NamedDimsStyle{S}}) where S
    inner_bc = unwrap_broadcasted(bc)
    copyto!(dest, inner_bc)
    L = unify_names(dimnames(dest), broadcasted_names(bc))
    return NamedDimsArray{L}(dest)
end

broadcasted_names(bc::Broadcasted) = broadcasted_names(bc.args...)
function broadcasted_names(a, bs...)
    a_name = broadcasted_names(a)
    b_name = broadcasted_names(bs...)
    unify_names_longest(a_name, b_name)
end
broadcasted_names(a::AbstractArray) = dimnames(a)
broadcasted_names(a) = tuple()
