"""
    NamedDimsArray{L, T, N, A}(data)

A `NamedDimsArray` is a wrapper array type, that provides a view onto  the
orignal array, which can have its dimensions refer to name rather than by
position.

For example:
```
xs = NamedDimsArray{(:features, :observations)}(data);

n_obs = size(xs, :observations)
feature_totals = sum(xs; dims=:observations)

first_obs_vector = xs[observations=1]
x = x[observations=15, features=2]  # 2nd feature in 15th observation.
```

`NamedDimsArray`s are normally a (near) zero-cost abstraction.
They generally resolve most dimension name related operations at compile
time.


The `NamedDimsArray` constructor takes a list of names as `Symbol`s,
one per dimension, and an array to wrap.
If the array being wrapped is a `NamedDimsArray` already then the new names
are combined with the existing names -- to replace wildcards (`:_`).
This will throw an error if the new names are not compatible with the old names
(i.e. if the nonwildcards do not match).
To assign new names to a `NamedDimsArray` without regard to compatibility with the old names
see `rename`(@ref).
"""
struct NamedDimsArray{L, T, N, A<:AbstractArray{T, N}} <: AbstractArray{T, N}
    # `L` is for labels, it should be an `NTuple{N, Symbol}`
    data::A
end

function NamedDimsArray{L}(orig::AbstractArray{T, N}) where {L, T, N}
    if !(L isa NTuple{N, Symbol})
        throw(ArgumentError(
            "A $N dimensional array, needs a $N-tuple of dimension names. Got: $L"
        ))
    end
    return NamedDimsArray{L, T, N, typeof(orig)}(orig)
end

@inline NamedDimsArray(orig::AbstractArray, names::Tuple) = NamedDimsArray{names}(orig)
@inline NamedDimsArray(orig::AbstractVector, name::Symbol) = NamedDimsArray(orig, (name,))

# Name-asserting constructor (like renaming, but only for wildcards (`:_`).)
NamedDimsArray{L}(orig::NamedDimsArray{L}) where L = orig
function NamedDimsArray{L}(orig::NamedDimsArray{old_names, T, N, A}) where {L, old_names, T, N, A}
    new_names = unify_names(L, old_names)
    return NamedDimsArray{new_names, T, N, A}(parent(orig))
end

parent_type(::Type{<:NamedDimsArray{L, T, N, A}}) where {L, T, N, A} = A

Base.parent(x::NamedDimsArray) = getfield(x, :data)

"""
    unname(A::NamedDimsArray) -> AbstractArray
    unname(A::AbstractArray) -> AbstractArray

Return the input array `A` without any dimension names.

For `NamedDimsArray`s this returns the parent array, equivalent to calling `parent`, but for
any other `AbstractArray` simply returns the input.
"""
unname(x::NamedDimsArray) = parent(x)
unname(x::AbstractArray) = x

"""
    dimnames(A) -> Tuple

Return the names of all the dimensions of the array `A`.
"""
dimnames(::Type{<:NamedDimsArray{L}}) where L = L
dimnames(::Type{<:AbstractArray{T, N}}) where {T, N} = ntuple(_->:_, N)
dimnames(x::T) where T<:AbstractArray = dimnames(T)

dim(a::NamedDimsArray{L}, name) where L = dim(L, name)


#############################
# AbstractArray Interface
# https://docs.julialang.org/en/v1/manual/interfaces/index.html#man-interface-array-1

## Minimal
Base.size(a::NamedDimsArray) = size(parent(a))
Base.size(a::NamedDimsArray, d) = size(parent(a), dim(a, d))


## optional
Base.IndexStyle(::Type{A}) where A<:NamedDimsArray = Base.IndexStyle(parent_type(A))

Base.length(a::NamedDimsArray) = length(parent(a))

Base.axes(a::NamedDimsArray) = axes(parent(a))
Base.axes(a::NamedDimsArray, d) = axes(parent(a), dim(a, d))


function named_size(a::AbstractArray{T,N}) where {T,N}
    L = dimnames(a)
    NamedTuple{L, NTuple{N,Int}}(size(a))
end
function Base.similar(
    a::NamedDimsArray{L,T},
    eltype::Type=T,
    dims::NamedTuple{new_names}=named_size(a)
) where {L,T,new_names}

    new_sizes = values(dims)
    return NamedDimsArray{new_names}(similar(parent(a), eltype, new_sizes))
end

function Base.similar(
    a::NamedDimsArray{L, T, N},
    eltype::Type,
    new_names::NTuple{N, Symbol}
) where {T,N,L}
    dims = NamedTuple{new_names, NTuple{N, Int}}(size(a))
    return similar(a, eltype, dims)
end

function Base.similar(
    a::NamedDimsArray{L, T, N},
    eltype::Type,
    new_sizes::NTuple{N, Int}
) where {L,T,N}

    dims = NamedTuple{L, NTuple{N, Int}}(new_sizes)
    return similar(a, eltype, dims)
end

###############################
# kwargs indexing

"""
    order_named_inds(A; named_inds...)

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
function order_named_inds(A::NamedDimsArray{L}; named_inds...) where {L}
    return order_named_inds(Val{L}(); named_inds...)
end

###################
# getindex / view / dotview
# Note that `dotview` is undocumented but needed for making `a[x=2] .= 3` work

for f in (:getindex, :view, :dotview)
    @eval begin
        @propagate_inbounds function Base.$f(A::NamedDimsArray; named_inds...)
            inds = order_named_inds(A; named_inds...)
            return Base.$f(A, inds...)
        end

        @propagate_inbounds function Base.$f(a::NamedDimsArray, inds::Vararg{<:Integer})
            # Easy scalar case, will just return the element
            return Base.$f(parent(a), inds...)
        end

        @propagate_inbounds function Base.$f(a::NamedDimsArray, ci::CartesianIndex)
            # Easy scalar case, will just return the element
            return Base.$f(parent(a), ci)
        end

        @propagate_inbounds function Base.$f(a::NamedDimsArray, inds...)
            # Some nonscalar case, will return an array, so need to give that names.
            data = Base.$f(parent(a), inds...)
            L = remaining_dimnames_from_indexing(dimnames(a), inds)
            return NamedDimsArray{L}(data)
        end
    end
end

############################################
# setindex!
@propagate_inbounds function Base.setindex!(a::NamedDimsArray, value; named_inds...)
    inds = order_named_inds(a; named_inds...)
    return setindex!(a, value, inds...)
end

@propagate_inbounds function Base.setindex!(a::NamedDimsArray, value, inds...)
    return setindex!(parent(a), value, inds...)
end
