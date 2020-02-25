# This file is for functions that just need simple standard overloading.

## Helpers:

function nameddimsarray_result(original_nda, reduced_data, reduction_dims)
    L = dimnames(original_nda)
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
            return NamedDimsArray{dimnames(a)}(data)
        end
    end
end

if VERSION > v"1.1-"
    function Base.eachslice(a::NamedDimsArray{L}; dims, kwargs...) where L
        numerical_dims = dim(a, dims)
        slices = eachslice(parent(a); dims=numerical_dims, kwargs...)
        return Base.Generator(slices) do slice
            # For unknown reasons (something to do with hoisting?) having this in the
            # function passed to `Generator` actually results in less memory being allocated
            names = remaining_dimnames_after_dropping(L, numerical_dims)
            return NamedDimsArray(slice, names)
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

for fun in (:(==), :isequal, :isapprox)
    @eval function Base.$fun(a::NamedDimsArray{La}, b::NamedDimsArray{Lb}; kw...) where {La, Lb}
        names_are_unifiable(La, Lb) || return false
        return $fun(parent(a), parent(b); kw...)
    end
end

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

# Two arrays
for (mod, funs) in (
    (:Base, (:sum!, :prod!, :maximum!, :minimum!)),
)
    for fun in funs
        @eval begin

            function $mod.$fun(a::NamedDimsArray{L}, b::AbstractArray) where L
                data = $mod.$fun(parent(a), b)
                return NamedDimsArray{L}(data)
            end

            function $mod.$fun(a::AbstractArray, b::NamedDimsArray{L}) where L
                data = $mod.$fun(a, parent(b))
                newL = unify_names_shortest(L, ntuple(_ -> :_, ndims(a)))
                return NamedDimsArray{newL}(data)
            end

            function $mod.$fun(a::NamedDimsArray{La}, b::NamedDimsArray{Lb}) where {La, Lb}
                newL = unify_names_shortest(La, Lb)
                data = $mod.$fun(parent(a), parent(b))
                return NamedDimsArray{newL}(data)
            end

        end
    end
end

Base.pop!(A::NamedDimsArray) = pop!(parent(A))
Base.popfirst!(A::NamedDimsArray) = popfirst!(parent(A))

function Base.append!(A::NamedDimsArray{L,T,1}, B::AbstractVector) where {L,T}
    newL = unify_names(L, dimnames(B))
    data = append!(parent(A), unname(B))
    return NamedDimsArray{newL}(data)
end

################################################
# map, collect

Base.map(f, A::NamedDimsArray) = NamedDimsArray(map(f, parent(A)), dimnames(A))

for (T, S) in [
    (:NamedDimsArray, :AbstractArray),
    (:AbstractArray, :NamedDimsArray),
    (:NamedDimsArray, :NamedDimsArray),
    ]
    for fun in [:map, :map!]

        # Here f::F where {F} is needed to avoid ambiguities in Julia 1.0
        @eval function Base.$fun(f::F, a::$T, b::$S, cs::AbstractArray...) where {F}
            data = $fun(f, unname(a), unname(b), unname.(cs)...)
            new_names = unify_names(dimnames(a), dimnames(b), dimnames.(cs)...)
            return NamedDimsArray(data, new_names)
        end

    end

    @eval function Base.foreach(f::F, a::$T, b::$S, cs::AbstractArray...) where {F}
        data = foreach(f, unname(a), unname(b), unname.(cs)...)
        unify_names(dimnames(a), dimnames(b), dimnames.(cs)...)
        return nothing
    end
end

Base.filter(f, A::NamedDimsArray{L,T,1}) where {L,T} = NamedDimsArray(filter(f, parent(A)), L)
Base.filter(f, A::NamedDimsArray{L,T,N}) where {L,T,N} = filter(f, parent(A))


# We overload collect on various kinds of `Generators` so that that can keep names.
function Base.collect(x::Base.Generator{<:NamedDimsArray{L}}) where {L}
    data = collect(Base.Generator(x.f, parent(x.iter)))
    return NamedDimsArray(data, L)
end

function Base.collect(x::Base.Generator{<:Iterators.Enumerate{<:NamedDimsArray{L}}}) where {L}
    data = collect(Base.Generator(x.f, enumerate(parent(x.iter.itr))))
    return NamedDimsArray(data, L)
end

Base.collect(x::Base.Generator{<:Iterators.ProductIterator{<:Tuple{<:NamedDimsArray,Vararg{Any}}}}) = collect_product(x)
Base.collect(x::Base.Generator{<:Iterators.ProductIterator{<:Tuple{<:Any,<:NamedDimsArray,Vararg{Any}}}}) = collect_product(x)
Base.collect(x::Base.Generator{<:Iterators.ProductIterator{<:Tuple{<:NamedDimsArray,<:NamedDimsArray,Vararg{Any}}}}) = collect_product(x)

function collect_product(x)
    data = collect(Base.Generator(x.f, Iterators.product(unname.(x.iter.iterators)...)))
    all_names = tuple_cat(dimnames.(x.iter.iterators)...)
    return NamedDimsArray(data, all_names)
end
