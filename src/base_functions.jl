

function Base.sum(f::Base.Callable, a::NamedDimsArray; dims)
    return sum(f, parent(a); dims=name2dim(A, dims))
end
