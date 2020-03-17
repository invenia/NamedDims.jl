# This file is for functions that explictly mess with the dimensions of a NameDimsArray

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
    @eval function $f(nda::NamedDimsArray{L}) where (L)
        data = $f(parent(nda))
        new_names = if ndims(nda) == 1 # vector input
            (:_, first(L))
        elseif ndims(data) == 1 # vector output
            (last(L),)
        else
            (last(L), first(L))
        end
        return NamedDimsArray{new_names}(data)
    end
end

# reshape
# For now we only implement the version that drops dimension names
Base.reshape(x::NamedDimsArray, d::Vararg{Union{Colon, Int}}) = reshape(parent(x), d)
