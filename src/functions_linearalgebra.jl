
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

# Need parent to explicitly call `getfield` to avoid infinitely calling `getproperty`
Base.parent(named::NamedFactorization) = getfield(named, :fact)
Base.size(named::NamedFactorization) = size(parent(named))

# Convenience constructors
for func in (:lu, :lu!, :lq, :lq!, :svd, :svd!, :qr, :qr!)
    @eval begin
        function LinearAlgebra.$func(nda::NamedDimsArray{L, T}, args...; kwargs...) where {L, T}
            fact = $func(parent(nda), args...; kwargs...)
            return NamedFactorization{L, T, typeof(fact)}(fact)
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
    if d === :U
        return NamedDimsArray{(n1,:_)}(inner)
    elseif d === :V
        return NamedDimsArray{(:_, n2)}(inner)
    elseif d === :Vt
        return NamedDimsArray{(n2,:_)}(inner)
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
        return NamedDimsArray{(n1, n1)}(inner)
    elseif F <: QRPivoted && d === :p
        return NamedDimsArray{(n1,)}(inner)
    else
        return inner
    end
end
