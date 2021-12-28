
#==
    Note on implementation of factorisition types:
    The general strategy is to create a named factorization type which wraps other factorizations.
    This allows us to define `getproperty` methods which return `NamedDimsArray`s around the
    corresponding properties of the parent factorization types.

    NOTE: Maybe we could just have a `Named` type that works with arrays and factorizations
    kind of like `Adjoint`?
==#

struct NamedFactorization{L, T, F<:Factorization{T}} <: Factorization{T}
    fact::F
end

function NamedFactorization{L}(fact::F) where {L, T, F <: Factorization{T}}
    return NamedFactorization{L,T,F}(fact)
end

# Need parent to explicitly call `getfield` to avoid infinitely calling `getproperty`
Base.parent(named::NamedFactorization) = getfield(named, :fact)
Base.size(named::NamedFactorization) = size(parent(named))
Base.propertynames(named::NamedFactorization; kwargs...) = propertynames(parent(named))

# Factorization type specific initial iterate calls
Base.iterate(named::NamedFactorization{L, T, <:LU}) where {L, T} = (named.L, Val(:U))
Base.iterate(named::NamedFactorization{L, T, <:LQ}) where {L, T} = (named.L, Val(:Q))
Base.iterate(named::NamedFactorization{L, T, <:SVD}) where {L, T} = (named.U, Val(:S))
function Base.iterate(
    named::NamedFactorization{L, T, <:Union{QR, LinearAlgebra.QRCompactWY, QRPivoted}}
) where {L, T}
    return (named.Q, Val(:R))
end

# Generic iterate follow up calls
function Base.iterate(named::NamedFactorization, st::Val{D}) where D
    r = iterate(parent(named), st)
    r === nothing && return nothing
    return (getproperty(named, D), last(r))
end

# Convenience constructors
for func in (:lu, :lu!, :lq, :lq!, :svd, :svd!, :qr, :qr!)
    @eval begin
        function LinearAlgebra.$func(nda::NamedDimsArray{L, T}, args...; kwargs...) where {L, T}
            return NamedFactorization{L}($func(parent(nda), args...; kwargs...))
        end
    end
end

# `getproperty` wrappers
## LU
function Base.getproperty(fact::NamedFactorization{L, T, <:LU}, d::Symbol) where {L, T}
    inner = getproperty(parent(fact), d)
    n1, n2 = L
    if d === :L
        return NamedDimsArray{(n1, :_)}(inner)
    elseif d === :U
        return NamedDimsArray{(:_, n2)}(inner)
    elseif d === :P
        perm_matrix_labels = (first(L), first(L))
        return NamedDimsArray{perm_matrix_labels}(inner)
    elseif d === :p
        perm_vector_labels = (first(L),)
        return NamedDimsArray{perm_vector_labels}(inner)
    else
        return inner
    end
end


## LQ
function Base.getproperty(fact::NamedFactorization{L, T, <:LQ}, d::Symbol) where {L, T}
    inner = getproperty(parent(fact), d)
    n1, n2 = L
    if d === :L
        return NamedDimsArray{(n1, :_)}(inner)
    elseif d === :Q
        return NamedDimsArray{(:_, n2)}(inner)
    else
        return inner
    end
end

## svd
function Base.getproperty(fact::NamedFactorization{L, T, <:SVD}, d::Symbol) where {L, T}
    inner = getproperty(parent(fact), d)
    n1, n2 = L
    # Naming based off the SVD visualization on wikipedia
    # # https://en.wikipedia.org/wiki/File:Singular_value_decomposition_visualisation.svg
    if d === :U
        return NamedDimsArray{(n1, :_)}(inner)
    elseif d === :V
        return NamedDimsArray{(n2, :_)}(inner)
    elseif d === :Vt
        return NamedDimsArray{(:_, n2)}(inner)
    else # :S
        return inner
    end
end

## qr
function Base.getproperty(
    fact::NamedFactorization{L, T, F},
    d::Symbol
) where {L, T, F<:Union{QR, LinearAlgebra.QRCompactWY, QRPivoted}}
    inner = getproperty(parent(fact), d)
    n1, n2 = L
    if d === :Q
        return NamedDimsArray{(n1,:_)}(inner)
    elseif d === :R
        return NamedDimsArray{(:_, n2)}(inner)
    elseif F <: QRPivoted && d === :P
        return NamedDimsArray{(n2, n2)}(inner)
    elseif F <: QRPivoted && d === :p
        return NamedDimsArray{(n2,)}(inner)
    else
        return inner
    end
end

function LinearAlgebra.:\(
    fact::NamedFactorization{L,T,F}, nda::NamedDimsArray{W}
) where {L,T,F<:Factorization{T},W}
    n1, n2 = L
    n1 != W[1] && throw(
        DimensionMismatch(
            "Mismatched dimensions with factorization: $L and NamedDimsArray: $W"
        ),
    )
    return NamedDimsArray{(n2,)}(LinearAlgebra.:\(parent(fact), parent(nda)))
end

function LinearAlgebra.:\(
    fact::NamedFactorization{L,T,F}, nda::AbstractVector
) where {L,T,F<:Factorization{T}}
    n1, n2 = L
    return NamedDimsArray{(n2,)}(LinearAlgebra.:\(parent(fact), nda))
end

# Specialised routines for \ often do in-place ops that result in the nameddim populated from B
# Leading to an incorrect named-dim
# We also unname here because that handles wrapper types
for S in (UpperTriangular, LowerTriangular)
    @eval begin
        function LinearAlgebra.:\(
            A::$S{T,<:NamedDimsArray{L}}, B::AbstractVector
        ) where {L,T}
            n1, n2 = L
            return NamedDimsArray{(n2,)}(LinearAlgebra.:\($S(unname(A)), parent(B)))
        end
    end
end

# Diagonal on a nameddim presently loses its nameddimsness. So just pass through for now.
LinearAlgebra.:\(A::Diagonal, B::NamedDimsArray) = LinearAlgebra.:\(A, parent(B))
