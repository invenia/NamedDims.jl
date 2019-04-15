# This file is for functions that explictly mess with the dimensions of a NameDimsArray

function Base.dropdims(nda::NamedDimsArray; dims)
    numerical_dims = dim(nda, dims)
    data = dropdims(parent(nda); dims=numerical_dims)
    L = remaining_dimnames_after_dropping(names(nda), numerical_dims)
    return NamedDimsArray{L}(data)
end
