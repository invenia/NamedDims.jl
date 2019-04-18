# This file is for functions that explictly mess with the dimensions of a NameDimsArray

function Base.dropdims(nda::NamedDimsArray; dims)
    numerical_dims = dim(nda, dims)
    data = dropdims(parent(nda); dims=numerical_dims)
    L = remaining_dimnames_after_dropping(names(nda), numerical_dims)
    return NamedDimsArray{L}(data)
end

# We use CoVector to workout if we are taking the tranpose of a tranpose etc
const CoVector = Union{Adjoint{<:Any, <:AbstractVector}, Transpose{<:Any, <:AbstractVector}}

for f in (:transpose, :adjoint, :permutedims)
    # Vector
    @eval function Base.$f(nda::NamedDimsArray{L,T,1}) where {L,T}
        new_names = (:_, first(L))
        return NamedDimsArray{new_names}($f(parent(nda)))
    end

    if f !==:permutedims
        # Vector Double Tranpose
        @eval function Base.$f(nda::NamedDimsArray{L,T,2,A}) where {L,T,A<:CoVector}
            new_names = (last(L),)  # drop the name of the first dimensions
            return NamedDimsArray{new_names}($f(parent(nda)))
        end
    end

    # Matrix
    @eval function Base.$f(nda::NamedDimsArray{L,T,2}) where {L,T}
        new_names = (last(L), first(L))
        return NamedDimsArray{new_names}($f(parent(nda)))
    end
end
