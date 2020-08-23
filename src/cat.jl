function Base.cat(a::NamedDimsArray{L}; dims) where L
    newL = expand_dimnames(L, dims)
    numerical_dims = dim(newL, dims)
    data = Base.cat(parent(a); dims=numerical_dims) # Base.cat is type unstable
    T = eltype(a)  # therefore the element type has to be inferred manually
    N = length(newL)  # as must the size of the array
    return NamedDimsArray{newL, T, N, Array{T,N}}(data)
end

function Base.cat(a::NamedDimsArray{L}, b::AbstractArray; dims) where L
    newL = expand_dimnames(L, dims)
    numerical_dims = dim(newL, dims)
    data = Base.cat(parent(a), b; dims=numerical_dims)
    T = promote_type(eltype(a), eltype(b))
    N = length(newL)
    return NamedDimsArray{newL, T, N, Array{T,N}}(data)
end

function Base.cat(a::AbstractArray, b::NamedDimsArray{L}; dims) where L
    newL = expand_dimnames(L, dims)
    numerical_dims = dim(newL, dims)
    data = Base.cat(a, parent(b); dims=numerical_dims)
    T = promote_type(eltype(a), eltype(b))
    N = length(newL)
    return NamedDimsArray{newL, T, N, Array{T,N}}(data)
end

function Base.cat(a::NamedDimsArray{La}, b::NamedDimsArray{Lb}; dims) where {La, Lb}
    combL = unify_names_shortest(La, Lb)
    newL = expand_dimnames(combL, dims)
    numerical_dims = dim(newL, dims)
    data = Base.cat(parent(a), parent(b); dims=numerical_dims)
    T = promote_type(eltype(a), eltype(b))
    N = length(newL)
    return NamedDimsArray{newL, T, N, Array{T,N}}(data)
end

# to dispatch on the first or the second argument being the NDA
for (T, S) in [
    (:NamedDimsArray, :AbstractArray),
    (:AbstractArray, :NamedDimsArray),
    (:NamedDimsArray, :NamedDimsArray)
    ]

    @eval function Base.cat(a::$T, b::$S, c::AbstractArray...; dims)
        return Base.cat(a, Base.cat(b, Base.cat(c...; dims=dims); dims=dims); dims=dims)
    end
end

# vcat and hcat
function Base.hcat(a::NamedDimsArray{L}) where L
    newL = expand_dimnames(L, 2)
    data = Base.hcat(parent(a))
    T = eltype(a)
    N = length(newL)
    return NamedDimsArray{newL, T, N, Array{T,N}}(data)
end

function Base.hcat(a::NamedDimsArray{L}) where L
    return Base.cat(a; dims=2)
end

Base.vcat(a::NamedDimsArray{L}) where L = a

# Base.hcat and Base.vcat specialise on this Union
const AbsVecOrMat = Union{AbstractVector, AbstractMatrix}
for (T, S) in [
    (:NamedDimsArray, :AbsVecOrMat),
    (:NamedDimsArray, :AbstractArray),
    (:AbsVecOrMat, :NamedDimsArray),
    (:AbstractArray, :NamedDimsArray),
    (:NamedDimsArray, :NamedDimsArray)
    ]

    for (fun, d) in zip((:vcat, :hcat), (1, 2))
        @eval function Base.$fun(a::$T, b::$S)
            combL = unify_names_shortest(dimnames(a), dimnames(b))
            newL = expand_dimnames(combL, $d)
            data = Base.$fun(parent(a), parent(b))
            T = promote_type(eltype(a), eltype(b))
            N = length(newL)
            return NamedDimsArray{newL, T, N, Array{T,N}}(data)
        end

        @eval function Base.$fun(a::$T, b::$S, c::NamedDimsArray...)
            return Base.cat(a, b, c...; dims=$d)
        end
    end
end
