# This supports nonbroadcasting math on NamedDimArrays

# Matrix product
valid_matmul_dims(a::Tuple{Symbol}, b::Tuple{Vararg{Symbol}}) = true
function valid_matmul_dims(a::Tuple{Symbol, Symbol}, b::Tuple{Vararg{Symbol}})
    a_dim = a[end]
    b_dim = b[1]

    return a_dim === b_dim || a_dim === :_ || b_dim === :_
end

matmul_names((a1, a2)::Tuple{Symbol,Symbol}, (b,)::Tuple{Symbol}) = (a1,)
matmul_names((a1, a2)::Tuple{Symbol,Symbol}, (b1, b2)::Tuple{Symbol,Symbol}) = (a1, b2)
matmul_names((a1,)::Tuple{Symbol,},(b1,b2)::Tuple{Symbol,Symbol}) = (a1, b2)

function matrix_prod_names(a, b)
    # 0 Allocations. See `@btime (()-> matrix_prod_names((:foo, :bar),(:bar,)))()`
    valid_matmul_dims(a, b) || throw(DimensionMismatch(
            "Cannot take matrix product of arrays with different inner dimension names. $a vs $b"
        ))
    res = matmul_names(a, b)
    compile_time_return_hack(res)
end

for (NA, NB) in ((1,2), (2,1), (2,2))  #Vector * Vector, is not allowed
    @eval function Base.:*(a::NamedDimsArray{A,T,$NA}, b::NamedDimsArray{B,S,$NB}) where {A,B,T,S}
        L = matrix_prod_names(A,B)
        data = *(parent(a), parent(b))
        return NamedDimsArray{L}(data)
    end

    @eval function Base.:*(a::NamedDimsArray{L,T,$NA}, b::AbstractArray{S,$NB}) where {L,T,S}
        return *(a, NamedDimsArray{names(b)}(b))
    end
    @eval function Base.:*(a::AbstractArray{T,$NA}, b::NamedDimsArray{L,S,$NB}) where {L,T,S}
        return *(NamedDimsArray{names(a)}(a), b)
    end
end
