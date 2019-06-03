
#==
    Note on implementation of factorisition types:
    The general strategy is to make one of the fields of the factorization types
    into a NamedDimsArray.
    We can then dispatch on this as required, and strip it out with `parent` for accessing operations.
    However, the type of the field does not actually always match to what it should be
    from a mathematical perspective.
    Which is corrected in `Base.getproperty`, as required
==#

## lu

function LinearAlgebra.lu!(nda::NamedDimsArray{L}, args...; kwargs...) where L
    inner_lu = lu!(parent(nda), args...; kwargs...)
    factors = NamedDimsArray{L}(getfield(inner_lu, :factors))
    ipiv = getfield(inner_lu, :ipiv)
    info = getfield(inner_lu, :info)
    return LU(factors, ipiv, info)
end

function Base.parent(fact::LU{T,<:NamedDimsArray{L}}) where {T, L}
    factors = parent(getfield(fact, :factors))
    ipiv = getfield(fact, :ipiv)
    info = getfield(fact, :info)
    return LU(factors, ipiv, info)
end

function Base.getproperty(fact::LU{T,<:NamedDimsArray{L}}, d::Symbol) where {T, L}
    inner = getproperty(parent(fact), d)
    n1, n2 = L
    if d == :L
        return NamedDimsArray{(n1, :_)}(inner)
    elseif d == :U
        return NamedDimsArray{(:_, n2)}(inner)
    elseif d == :P
        perm_matrix_labels = (first(L), first(L))
        return NamedDimsArray{perm_matrix_labels}(inner)
    elseif d == :p
        perm_vector_labels = (first(L),)
        return NamedDimsArray{perm_vector_labels}(inner)
    else
        return inner
    end
end


## lq

LinearAlgebra.lq(nda::NamedDimsArray, args...; kws...) = lq!(copy(nda), args...; kws...)
function LinearAlgebra.lq!(nda::NamedDimsArray{L}, args...; kwargs...) where L
    inner = lq!(parent(nda), args...; kwargs...)
    factors = NamedDimsArray{L}(getfield(inner, :factors))
    τ = getfield(inner, :τ)
    return LQ(factors, τ)
end

function Base.parent(fact::LQ{T,<:NamedDimsArray{L}}) where {T, L}
    factors = parent(getfield(fact, :factors))
    τ = getfield(fact, :τ)
    return LQ(factors, τ)
end

function Base.getproperty(fact::LQ{T,<:NamedDimsArray{L}}, d::Symbol) where {T, L}
    inner = getproperty(parent(fact), d)
    n1, n2 = L
    if d == :L
        return NamedDimsArray{(n1, :_)}(inner)
    elseif d == :Q
        return NamedDimsArray{(:_, n2)}(inner)
    else
        return inner
    end
end

## svd

function LinearAlgebra.svd(nda::NamedDimsArray{L, T}, args...; kwargs...) where {L, T}
    return svd!(
        LinearAlgebra.copy_oftype(nda, LinearAlgebra.eigtype(T)),
        args...;
        kwargs...
    )
end

function LinearAlgebra.svd!(nda::NamedDimsArray{L}, args...; kwargs...) where L
    inner = svd!(parent(nda), args...; kwargs...)
    u = NamedDimsArray{L}(getfield(inner, :U))
    s = getfield(inner, :S)
    vt = NamedDimsArray{L}(getfield(inner, :Vt))
    return SVD(u, s, vt)
end

function Base.parent(fact::SVD{T, Tr, <:NamedDimsArray{L}}) where {T, Tr, L}
    u = parent(getfield(fact, :U))
    s = getfield(fact, :S)
    vt = parent(getfield(fact, :Vt))
    return SVD(u, s, vt)
end

function Base.getproperty(fact::SVD{T, Tr, <:NamedDimsArray{L}}, d::Symbol) where {T, Tr, L}
    inner = getproperty(parent(fact), d)
    n1, n2 = L
    if d == :U
        return NamedDimsArray{(n1,:_)}(inner)
    elseif d == :V
        return NamedDimsArray{(:_, n2)}(inner)
    elseif d == :Vt
        return NamedDimsArray{(n2,:_)}(inner)
    else # :S
        return inner
    end
end
