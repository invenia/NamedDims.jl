
struct NamedDimsArray{L<:Tuple, T, N, A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end

function NamedDimsArray(orig::AbstractArray{T,N}, names) where {T, N}
    if length(names) != N
        throw(ArgumentError("A $N dimentional array, need $N dimension names. Got: $names"))
    end
    names_tt = Tuple{names...}
    return NamedDimsArray{names_tt, T, N, typeof(orig)}(orig)
end


Base.parent(x::NamedDimsArray) = x.data


"""
    dim_names(A)

Returns a tuple of containing the names of all the dimensions of the array `A`.
"""
dim_names(::Type{<:NamedDimsArray{L}}) where L = Tuple(L.parameters)
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
function order_named_inds(A; named_inds...)
    keys(named_inds) ⊆ dim_names(A) || throw(
        DimensionMismatch("Expected $(dim_names(A)), got $(keys(named_inds))")
    )


    inds = map(dim_names(A)) do name
        get(named_inds, name, :)  # default to slicing
    end
end

function Base.getindex(A::NamedDimsArray; named_inds...)
    inds = order_named_inds(A; named_inds...)
    return getindex(parent(A), inds...)
end


function Base.setindex(A::NamedDimsArray, value; named_inds...)
    inds = order_named_inds(A; named_inds...)
    return setindex(parent(A), value, inds...)
end
