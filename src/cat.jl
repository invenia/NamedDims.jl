function Base.cat(a::NamedDimsArray{L}; dims) where L
    newL = expand_dimnames(L, dims)
    numerical_dims = dim(newL, dims)
    data = Base.cat(parent(a); dims=numerical_dims) # Base.cat is type unstable
    return NamedDimsArray{newL}(data)
end

# While the following two functions are covered by the general case below where splatted c
# is empty, they are included because they reduce allocations to that of regular arrays.
function Base.cat(a::NamedDimsArray{L}, b::AbstractArray; dims) where L
    newL = expand_dimnames(L, dims)
    numerical_dims = dim(newL, dims)
    data = Base.cat(parent(a), b; dims=numerical_dims)
    return NamedDimsArray{newL, eltype(data), ndims(data), typeof(data)}(data)
end

function Base.cat(a::AbstractArray, b::NamedDimsArray{L}; dims) where L
    newL = expand_dimnames(L, dims)
    numerical_dims = dim(newL, dims)
    data = Base.cat(a, parent(b); dims=numerical_dims)
    T = promote_type(eltype(a), eltype(b))
    N = length(newL)
    return NamedDimsArray{newL, eltype(data), ndims(data), typeof(data)}(data)
end

# to dispatch on the first _or the second_ argument being the NDA.
for (T, S) in [
    (:NamedDimsArray, :AbstractArray),
    (:AbstractArray, :NamedDimsArray),
    (:NamedDimsArray, :NamedDimsArray)
    ]

    @eval function Base.cat(a::$T, b::$S, cs::AbstractArray...; dims)
        combL = unify_names_longest(dimnames(a), dimnames(b), dimnames.(cs)...)
        newL = expand_dimnames(combL, dims)
        numerical_dims = dim(newL, dims)
        data = Base.cat(unname(a), unname(b), unname.(cs)...; dims=numerical_dims)
        return NamedDimsArray{newL}(data)
    end
end

function Base.hcat(a::NamedDimsArray{L}) where L
    newL = expand_dimnames(L, 2)
    data = Base.hcat(parent(a))
    return NamedDimsArray{newL}(data)
end

Base.vcat(a::NamedDimsArray{L}) where L = a

for (T, S) in [
    (:NamedDimsVecOrMat, :NamedDimsVecOrMat),
    (:NamedDimsVecOrMat, :AbstractVecOrMat),
    (:AbstractVecOrMat, :NamedDimsVecOrMat),
    ]
    for (fun, d) in zip((:vcat, :hcat), (1, 2))

        @eval function Base.$fun(a::$T, b::$S, cs::AbstractVecOrMat...)
            combL = unify_names_longest(dimnames(a), dimnames(b), dimnames.(cs)...)
            newL = expand_dimnames(combL, $d)
            data = Base.$fun(unname(a), unname(b), unname.(cs)...)
            return NamedDimsArray{newL}(data)
        end
    end
end

