# This file is for functions that explictly mess with the dimensions of a NameDimsArray

function Base.dropdims(nda::NamedDimsArray; dims)
    numerical_dims = dim(nda, dims)
    data = dropdims(parent(nda); dims=numerical_dims)
    L = remaining_dimnames_after_dropping(names(nda), numerical_dims)
    return NamedDimsArray{L}(data)
end


# Vector
function Base.adjoint(nda::NamedDimsArray{L, T, 1}) where {L,T}
    new_names = (:_, first(L))
    return NamedDimsArray{new_names}(adjoint(parent(nda)))
end

# Vector Double
function Base.adjoint(nda::NamedDimsArray{L, T, 2, A}) where {L,T, A<:LinearAlgebra.Adjoint{T, <:AbstractVector}}

    new_names = (last(L),)  # drop the name of the first dimensions
    return NamedDimsArray{new_names}(adjoint(parent(nda)))
end

# Matrix
function Base.adjoint(nda::NamedDimsArray{L, T, 2}) where {L,T}
    new_names = (last(L), first(L))
    return NamedDimsArray{new_names}(adjoint(parent(nda)))
end
