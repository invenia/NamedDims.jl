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
    for func in (:(Base.:*), :(LinearAlgebra.lmul!))
        @eval function $func(a::NamedDimsArray{A,T,$NA}, b::NamedDimsArray{B,S,$NB}) where {A,B,T,S}
            L = matrix_prod_names(A,B)
            data = $func(parent(a), parent(b))
            return NamedDimsArray{L}(data)
        end
    end
end

"""
    @declare_matmul(MatrixT, VectorT=nothing)

This macro helps define matrix multiplication for the types
with 2D type parameterization `MatrixT` and 1D `VectorT`.
It defines the various overloads for `Base.:*` that are required.
It should be used at the top level of a module.
"""
macro declare_matmul(MatrixT, VectorT=nothing)
    codes = Expr(:block)
    dim_combos = VectorT === nothing ? ((2,2),) : ((1,2), (2,1), (2,2))

    for (NA, NB) in dim_combos
        TA_named = :(NamedDims.NamedDimsArray{<:Any, <:Any, $NA})
        TB_named = :(NamedDims.NamedDimsArray{<:Any, <:Any, $NB})
        TA_other = (VectorT, MatrixT)[NA]
        TB_other = (VectorT, MatrixT)[NB]
        for func in (:(Base.:*), :(LinearAlgebra.lmul!))
            code = quote
                function $func(a::$TA_named, b::$TB_other)
                    return $func(a, NamedDims.NamedDimsArray{NamedDims.names(b)}(b))
                end
                function $func(a::$TA_other, b::$TB_named)
                    return $func(NamedDims.NamedDimsArray{NamedDims.names(a)}(a), b)
                end
            end
            push!(codes.args, code)
        end
    end
    return esc(codes)
end

@declare_matmul(AbstractMatrix, AbstractVector)
@declare_matmul(Diagonal,)
