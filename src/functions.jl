# This file is for functions that just need simple standard overloading.

## Helpers:

function nameddimsarray_result(original_nda, reduced_data, reduction_dims)
    L = names(original_nda)
    return NamedDimsArray{L}(reduced_data)
end

# if reducing over `:` then results is a scalar
function nameddimsarray_result(original_nda, reduced_data, reduction_dims::Colon)
    return reduced_data
end

################################################
# Overloads

# 1 Arg
for (mod, funs) in (
    (:Base, (:sum, :prod, :maximum, :minimum, :extrema)),
    (:Statistics, (:mean, :std, :var, :median)),
)
    for fun in funs
        @eval function $mod.$fun(a::NamedDimsArray; dims=:, kwargs...)
            numerical_dims = dim(a, dims)
            data = $mod.$fun(parent(a); dims=numerical_dims, kwargs...)
            return nameddimsarray_result(a, data, numerical_dims)
        end
    end
end

# 1 Arg - no default for `dims` keyword
for (mod, funs) in (
    (:Base, (:cumsum, :cumprod, :sort, :sort!)),
)
    for fun in funs
        @eval function $mod.$fun(a::NamedDimsArray; dims, kwargs...)
            numerical_dims = dim(a, dims)
            data = $mod.$fun(parent(a); dims=numerical_dims, kwargs...)
            return nameddimsarray_result(a, data, numerical_dims)
        end

        # Vector case
        @eval function $mod.$fun(a::NamedDimsArray{L, T, 1}; kwargs...) where {L, T}
            data = $mod.$fun(parent(a); kwargs...)
            return NamedDimsArray{NamedDims.names(a)}(data)
        end
    end
end

# 1 arg before - no default for `dims` keyword
for (mod, funs) in (
    (:Base, (:mapslices,)),
)
    for fun in funs
        @eval function $mod.$fun(f, a::NamedDimsArray; dims, kwargs...)
            numerical_dims = dim(a, dims)
            data = $mod.$fun(f, parent(a); dims=numerical_dims, kwargs...)
            return nameddimsarray_result(a, data, numerical_dims)
        end
    end
end

# 2 arg before
for (mod, funs) in (
    (:Base, (:mapreduce,)),
)
    for fun in funs
        @eval function $mod.$fun(f1, f2, a::NamedDimsArray; dims=:, kwargs...)
            numerical_dims = dim(a, dims)
            data = $mod.$fun(f1, f2, parent(a); dims=numerical_dims, kwargs...)
            return nameddimsarray_result(a, data, numerical_dims)
        end
    end
end

################################################
# Non-dim Overloads

# Array then perhaps other args
for (mod, funs) in (
    (:Base, (:zero, :one, :copy, :empty!, :push!, :pushfirst!)),
)
    for fun in funs
        @eval function $mod.$fun(a::NamedDimsArray{L}, x...) where L
            data = $mod.$fun(parent(a), x...)
            return NamedDimsArray{L}(data)
        end
    end
end

Base.pop!(A::NamedDimsArray) = pop!(parent(A))
Base.popfirst!(A::NamedDimsArray) = popfirst!(parent(A))

function Base.append!(A::NamedDimsArray{L,T,1}, B::AbstractVector) where {L,T}
    newL = unify_names(L, names(B))
    data = append!(parent(A), unname(B))
    return NamedDimsArray{newL}(data)
end

################################################
# Generators

if VERSION > v"1.1-"
    Base.eachslice(A::NamedDimsArray; dims) = _eachslice(A, dims)
else
    eachcol(A::AbstractVecOrMat) = (view(A, :, i) for i in axes(A, 2))
    eachrow(A::AbstractVecOrMat) = (view(A, i, :) for i in axes(A, 1))
    # every line identical to Base, but no _eachslice(A, dims) to disatch on.
    eachslice(A::AbstractArray; dims) = _eachslice(A, dims)
end

function _eachslice(A::AbstractArray, dims::Symbol)
    numerical_dims = dim(A, dims)
    return _eachslice(A, numerical_dims)
end
function _eachslice(A::AbstractArray, dims::Tuple)
    length(dims) == 1 || throw(ArgumentError("only single dimensions are supported"))
    return _eachslice(A, first(dims))
end
@inline function _eachslice(A::AbstractArray, dim::Int)
    dim <= ndims(A) || throw(DimensionMismatch("A doesn't have $dim dimensions"))
    idx1, idx2 = ntuple(d->(:), dim-1), ntuple(d->(:), ndims(A)-dim)
    return (view(A, idx1..., i, idx2...) for i in axes(A, dim))
end

function Base.collect(itr::Base.Generator{<:NamedDimsArray{L}}) where {L}
    NamedDimsArray{L}(collect(Base.Generator(itr.f, parent(itr.iter))))
end
