
# `L` is for labels, it should be a `Tuple` of `Symbol`s
struct NamedDimsArray{L, T, N, A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end

function NamedDimsArray(orig::AbstractArray{T,N}, names::NTuple{N, Symbol}) where {T, N}
    return NamedDimsArray{names, T, N, typeof(orig)}(orig)
end


Base.parent(x::NamedDimsArray) = x.data


"""
    dim_names(A)

Returns a tuple of containing the names of all the dimensions of the array `A`.
"""
dim_names(::Type{<:NamedDimsArray{L}}) where L = L
dim_names(x::T) where T = dim_names(T)


name2dim(a::NamedDimsArray{L}, name) where L = name2dim(L, name)



#############################
# AbstractArray Interface
# https://docs.julialang.org/en/v1/manual/interfaces/index.html#man-interface-array-1

Base.size(a::NamedDimsArray) = size(parent(a))
Base.getindex(A::NamedDimsArray, inds...) = getindex(parent(A), inds...)
Base.setindex(A::NamedDimsArray, value, inds...) = setindex(parent(A), value, inds...)


###############################
# kwargs indexing

"""
    order_named_inds(A, named_inds...)

Returns the indices that have the names and values given by `named_inds`
sorted into the order expected for the dimension of the array `A`.
If any dimensions of `A` are not present in the named_inds,
then they are given the value `:`, for slicing

For example:
```
A = NamedDimArray(rand(4,4), (:x,, :y))
order_named_inds(A; y=10, x=13) == (13,10)
order_named_inds(A; x=2, y=1:3) == (2, 1:3)
order_named_inds(A; y=5) == (:, 5)
```

This provides the core indexed lookup for `getindex` and `setindex` on the Array `A`
"""
order_named_inds(A::AbstractArray; named_inds...) = order_named_inds(dim_names(A); named_inds...)

function Base.getindex(A::NamedDimsArray; named_inds...)
    inds = order_named_inds(A; named_inds...)
    return getindex(parent(A), inds...)
end


function Base.setindex(A::NamedDimsArray, value; named_inds...)
    inds = order_named_inds(A; named_inds...)
    return setindex(parent(A), value, inds...)
end
