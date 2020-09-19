function Base.cat(a::NamedDimsArray{L}; dims) where L
    newL = expand_dimnames(L, dims)
    numerical_dims = dim(newL, dims)
    data = Base.cat(parent(a); dims=numerical_dims) # Base.cat is type unstable
    return NamedDimsArray{newL}(data)
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

for (f, nf, tf, tup) in [
    (:vcat, :_named_vcat, :_typed_vcat, ()),
    (:hcat, :_named_hcat, :_typed_hcat, (:_,)),
    ]
    @eval begin
        Base.reduce(::typeof($f), A::AbstractVector{<:NamedDimsVecOrMat}) =
            $nf(mapreduce(dimnames, unify_names_longest, A), A)
        Base.reduce(::typeof($f), A::NamedDimsVector{<:Any,<:AbstractVecOrMat}) =
            $nf(mapreduce(dimnames, unify_names_longest, A), A)
        Base.reduce(::typeof($f), A::NamedDimsVector{<:Any,<:NamedDimsVecOrMat}) =
            $nf(mapreduce(dimnames, unify_names_longest, A), A)

        function $nf(Linner, A)
            Louter = ($tup..., dimnames(A)...)
            Lnew = unify_names_longest(Linner, Louter)
            data = Base.$tf(mapreduce(eltype, promote_type, A), A) # same as Base
            return NamedDimsArray{Lnew}(data)
        end
    end
end

