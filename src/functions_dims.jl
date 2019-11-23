# This file is for functions that explictly mess with the dimensions of a NameDimsArray

"""
    rename(nda::NamedDimsArray, names)

Returns a new `NameDimsArray` with the given dimension `names`.
`rename` outright replaces the names; while still wrapping the same backing array.
Unlike the constructor, it does not require that new names are compatible
with the old names (though you do still need to match the number of dimensions).
"""
rename(nda::NamedDimsArray, names) = NamedDimsArray(parent(nda), names)

function Base.dropdims(nda::NamedDimsArray; dims)
    numerical_dims = dim(nda, dims)
    data = dropdims(parent(nda); dims=numerical_dims)
    L = remaining_dimnames_after_dropping(dimnames(nda), numerical_dims)
    return NamedDimsArray{L}(data)
end

function Base.permutedims(nda::NamedDimsArray{L}, perm) where {L}
    numerical_perm = dim(nda, perm)
    new_names = permute_dimnames(L, numerical_perm)

    return NamedDimsArray{new_names}(permutedims(parent(nda), numerical_perm))
end

function Base.selectdim(nda::NamedDimsArray, s::Symbol, i)
    return selectdim(nda, dim(nda, s), i)
end

for f in (
    :(Base.transpose),
    :(Base.adjoint),
    :(Base.permutedims),
    :(LinearAlgebra.pinv)
)
    # Vector
    @eval function $f(nda::NamedDimsArray{L,T,1}) where {L,T}
        new_names = (:_, first(L))
        return NamedDimsArray{new_names}($f(parent(nda)))
    end

    # Vector Double Transpose
    if f != :(Base.permutedims)
        @eval function $f(nda::NamedDimsArray{L,T,2,A}) where {L,T,A<:CoVector}
            new_names = (last(L),)  # drop the name of the first dimensions
            return NamedDimsArray{new_names}($f(parent(nda)))
        end
    end

    # Matrix
    @eval function $f(nda::NamedDimsArray{L,T,2}) where {L,T}
        new_names = (last(L), first(L))
        return NamedDimsArray{new_names}($f(parent(nda)))
    end
end


# reshape
# For now we only implement the version that drops dimension names
Base.reshape(x::NamedDimsArray, d::Vararg{Union{Colon, Int}}) = reshape(parent(x), d)
